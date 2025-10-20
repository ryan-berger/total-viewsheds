//! The main entrypoint for running computations.

use color_eyre::{eyre::Ok, Result};
use std::arch::x86_64::{
    __m128, __m256, _mm256_blend_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_max_ps,
    _mm256_set1_ps, _mm256_slli_si256, _mm_broadcast_ss, _mm_max_ps, _mm_set1_ps,
};
use std::iter::zip;
use std::mem::transmute;
use std::simd::prelude::*;
use std::thread;
use std::time::Instant;

/// Handles all the computations.
pub struct Compute<'compute> {
    /// Where to run the kernel computations
    backend: crate::config::Backend,
    /// What to compute.
    process: Vec<crate::config::Process>,
    /// Vulkan GPU manager
    vulkan: Option<super::vulkan::Vulkan>,
    /// The OS's state directory for saving our cache into.
    state_directory: Option<std::path::PathBuf>,
    /// Output directory
    output_directory: Option<std::path::PathBuf>,
    /// Storage interface for conputed ring (viewshed) data.
    storage: Option<crate::output::ring_data::Storage>,
    /// The Digital Elevation Model that we're computing.
    dem: &'compute mut crate::dem::DEM,
    /// The constants for each kernel computation.
    pub constants: kernel::constants::Constants,
    /// The amount of reserved memory for ring data.
    total_reserved_rings: usize,
    /// Keeps track of the cumulative surfaces from every angle.
    pub total_surfaces: Vec<f32>,
    /// Keeps track of the ring (viewshed) data.
    pub ring_data: Vec<Vec<u32>>,
    /// Keeps track of the longest lines of sight.
    pub longest_lines: Vec<f32>,
}

/// `generate_rotation` generates a rotation "map" for a given elevation list
/// Adapted from [this stack overflow answer](https://stackoverflow.com/a/71901621)
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "asdfasdf"
)]
fn generate_rotation(elevs: &[i16], angle: f64, max_los: usize) -> (Vec<i32>, Vec<i16>) {
    let width = (max_los * 3) as isize;

    assert_eq!(width % 2, 0);
    assert_eq!(elevs.len() as isize % width, 0);
    assert_eq!(elevs.len() as isize / width, width);

    let (sin, cos) = (f64::sin(angle.to_radians()), f64::cos(angle.to_radians()));
    let (x_center, y_center) = (width / 2, width / 2);

    let mut rotation = Vec::with_capacity(2 * max_los * max_los);

    for x in (max_los as isize)..(max_los as isize) * 2 {
        let x_sin = (x - x_center) as f64 * sin;
        let x_cos = (x - x_center) as f64 * cos;
        for y in (max_los as isize)..width {
            let y_sin = (y - y_center) as f64 * sin;
            let y_cos = (y - y_center) as f64 * cos;

            let x_rot = (x_cos - y_sin).round() as isize + y_center;
            let y_rot = (y_cos + x_sin).round() as isize + x_center;

            let new_idx = x_rot.clamp(0, width - 1) * width + y_rot.clamp(0, width - 1);
            let normalized = if new_idx >= 0 && new_idx < elevs.len() as isize {
                new_idx
            } else {
                unreachable!()
            };

            rotation.push(normalized as i32);
        }
    }

    assert_eq!(
        rotation.len() as isize,
        max_los as isize * (width - max_los as isize)
    );

    // map the indexes to their elevations
    let elevations = rotation
        .iter()
        .map(|&idx| {
            if idx < 0i32 {
                i16::MIN
            } else {
                // safety: idx is clamped so a get will always be in-bounds
                *unsafe { elevs.get_unchecked(idx as usize) }
            }
        })
        .collect::<Vec<i16>>();

    let idxs = (0..max_los)
        .flat_map(|idx| {
            let start = idx * (2 * max_los);
            let end = start + max_los;
            &rotation[start..end]
        })
        .map(|&val| {
            let x = (val / width as i32) - max_los as i32;
            let y = (val % width as i32) - max_los as i32;
            if (0i32..max_los as i32).contains(&x) && (0i32..max_los as i32).contains(&y) {
                x * (max_los as i32) + y
            } else {
                -1i32
            }
        })
        .collect();

    (idxs, elevations)
}

/// `EARTH_RADIUS_SQUARED` is the earth's radius squared in meters
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

/// `TAN_ONE_RAD` helps normalize the fact that inner points are sampled more often
/// see the TVS paper for reasoning.
const TAN_ONE_RAD: f32 = 0.017_453_3;

/// `total_viewshed` calculates a straight-lined total viewshed using AVX2 instructions
#[target_feature(enable = "avx2")]
fn total_viewshed_vector(elevation_map: &[i16], indexes: &[i32], max_los: usize, result: &mut [f32]) {
    assert_eq!(
        elevation_map.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    assert_eq!(
        max_los % 8,
        0,
        "to help the vectorizer, max_los must be a multiple of 8"
    );

    let width = 2 * max_los;

    // precalculate all distances and their spherical earth "adjustments".
    // This saves ~33% of effort inside our hot loop
    let distances = (0..max_los)
        .map(|x| {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "x is in [1..max_los), max_los < 2^23"
            )]
            let distance = (x * 100) as f32;
            (distance, distance / EARTH_RADIUS_SQUARED)
        })
        .collect::<Vec<(f32, f32)>>();

    // allocate angles and their prefix maxes as mm256s so that the underlying buffer
    // is correctly aligned to 256 bits
    let mut angles: Vec<__m256> = vec![_mm256_set1_ps(0.0); max_los / 8];
    let mut prefix_max: Vec<__m256> = vec![_mm256_set1_ps(0.0); max_los / 8];

    // constant lowest value spatted across all 8 lanes
    let max_angles = _mm256_set1_ps(-2000.0f32);

    for line_idx in 0..max_los {
        let elevation_offset = line_idx * width;
        #[expect(
            clippy::indexing_slicing,
            reason = "elevation_offset < (elevation_map.len()-max_los) so slicing is always in bounds"
        )]
        let line = &elevation_map[elevation_offset..(elevation_offset + width)];

        let indexes_offset = line_idx * max_los;
        #[expect(
            clippy::indexing_slicing,
            reason = "index_offset < (indexes.len()-max_los) so slicing is always in bounds"
        )]
        let line_indexes = &indexes[indexes_offset..(indexes_offset + max_los)];

        // The hottest of the hot loops.
        // Any change inside this loop needs careful benchmarking before committing
        for pov in 0..max_los {
            #[expect(
                clippy::indexing_slicing,
                reason = "line_indexes is max_los long so pov is always in bounds"
            )]
            let result_idx = line_indexes[pov];

            // if the line of sight is not within our computable points, do not consider it
            if result_idx < 0i32 {
                continue;
            }

            // safety: pov is guaranteed to be in bounds since the slice is max_los in size
            let pov_height = f32::from(unsafe { *line.get_unchecked(pov) });

            // convert the max_los-1 elevations ahead of the POV into floats, and adjust
            // for the observer's height
            #[expect(
                clippy::indexing_slicing,
                reason = "[pov+1, pov+max_los) is always in bounds"
            )]
            let elevations = line[pov..pov + max_los]
                .iter()
                .map(|&x| f32::from(x) - pov_height);

            // safety: sizeof(__m256) ==  sizeof(f32) * 8, meaning it is well-aligned
            let flat_angles = unsafe { transmute::<&mut [__m256], &mut [f32]>(angles.as_mut()) };

            zip(flat_angles.iter_mut(), &distances)
                .zip(elevations)
                .for_each(|((angle, (distance, adjustment)), elevation)| {
                    *angle = (elevation / distance) - adjustment;
                });

            // get rid of NaN in the first angle calculation since there is a division by zero
            flat_angles[0] = -2001.0;

            // carry out a SIMD prefix max using AVX instructions a la https://en.algorithmica.org/hpc/algorithms/prefix/
            // This method is a bit inefficient from an algorithmic perspective, as we are doing n*log(n) amount of work,
            // but in the end, the ability to make use of AVX makes up for this, and doubles the speed on benchmarks.
            //
            // First we compute a prefix max for blocks of 4 f32s, but 8 at a time
            // (like `prefix` in algorithmica's algorithm).
            // Each AVX vector register has 8 lanes, but are in groups of 4:
            // angle_vec = | a_1 | a_2 | a_3 | a_4 | b_1 | b_2 | b_3 | b_4 |
            //
            // Operations such as max_ps operate on 128bit chunks, but two at a time.
            //
            // To do start we shift the lanes left by 4 bytes in an intermediary register leaving zeros:
            // | 0 | a_1 | a_2 | a_3 | 0 | b_1 | b_2 | b_3 |
            //
            // Having zero shifted in is helpful if you are doing a prefix sum since 0 added to anything is itself,
            // an identity element. However, zero is not a good identity element for `fmax`. Instead, we use
            // -2000.0, which is lower than any angle calculation that we'll ever do making sure that
            // ident(x) = fmax(x, -2000.0) = x.
            //
            // We blend in a constant vector full of -2000.0s, via blend_ps, passing in a mask
            // of 0b1000_1000 which makes sure to only blend the first elements:
            // shifted_vec = | -2000.0 | a_1 | a_2 | a_3 | -2000.0 | b_1 | b_2 | b_3 |
            //
            // And finally:
            // v_prefix_max = max_ps(angle_vec, shifted_vec)
            // = max (|   a_1   | a_2 | a_3 | a_4 |   b_1   | b_2 | b_3 | b_4 |,
            //        | -2000.0 | a_1 | a_2 | a_3 | -2000.0 | b_2 | b_3 | b_4 |)
            // =      |   a_1   | max(a_1, a_2) | max(a_2, a_3) | ...
            //
            // Visually, it should be clear that we have only computed the prefix max for the
            // first two out of four elements. We can repeat this trick a second time but this time
            // shifting our prefix max twice (and blending with a mask of 1100_1100),
            // fully computing the prefix sum for both blocks of 4 elements.
            // For brevity, only the first 4 lanes are pictured:
            //
            // v_prefix_max = |   a_1   | max(a_1, a_2) | max(a_2, a_3) | max(a_3, a_4) |
            // shifted_vec =  | -2000.0 |    -2000.0    | a_1           | max(a_1, a_2) |
            //
            // v_prefix_max = max_ps(v_prefix_max, shifted_vec)
            // = | a_1 | max(a_1, a_2) | max(a_1, max(a_2, a_3)) | max(max(a_1, a_2), max(a_3, a_4))
            //
            // MAGICAL
            //
            // However, now we have blocks of 4 prefix maxes calculated, not a prefix max calculated
            // for the full array. To do so, we need to make a second pass to accumulate the
            // results across blocks. Doing this is fairly simple. Take the last element of
            // each block and "splat" it across all lanes:
            //
            // highest_from_block = | cur_block[3] | cur_block[3] | cur_block[3] | cur_block[3] |
            //
            // Our global max is 4 lanes of the max of all previous maxes of all other blocks
            // In other words, it is an accumulated max:
            //
            // global_max = | x | x | x | x |
            //
            // Our new current block needs to be updated with the computation from the global_max,
            // so just re-compute the current block by taking the max of it and global_max,:
            //
            // cur_block = max_ps(cur_block, global_max)
            //
            // and our new global_max is a scalar spatted across all lanes of the cumulative
            // maximum of all other blocks:
            //
            // global_max = max_px(highest_from_block, global_max)
            //
            // And there you have it!
            //
            // To check to see if a point is visible, we can check whether the point is greater
            // than or equal to the current prefix max:
            //
            // visibile(index) = angle[index] >= prefix_max[index]

            // Calculate the 4-wide block prefix max two at a time
            for (prefix, &angle) in zip(&mut prefix_max, &angles) {
                let mut v_prefix_max: __m256 = {
                    let shifted = _mm256_slli_si256::<4>(_mm256_castps_si256(angle));
                    let blended =
                        _mm256_blend_ps::<0b1000_1000>(_mm256_castsi256_ps(shifted), max_angles);

                    _mm256_max_ps(angle, blended)
                };

                v_prefix_max = {
                    let shifted = _mm256_slli_si256::<8>(_mm256_castps_si256(v_prefix_max));
                    let blended =
                        _mm256_blend_ps::<0b1100_1100>(_mm256_castsi256_ps(shifted), max_angles);
                    _mm256_max_ps(v_prefix_max, blended)
                };

                *prefix = v_prefix_max;
            }

            // safety: sizeof(__m256) ==  sizeof(__m128) * 2, meaning it is well-aligned
            let single_wide_angles =
                unsafe { transmute::<&mut [__m256], &mut [__m128]>(prefix_max.as_mut()) };

            let mut acc = _mm_set1_ps(-2000.0f32);

            // accumulate the prefix maxes for blocks, re-computing all prefix maxes
            // to include the accumulated value
            for prefix in single_wide_angles {
                let cur_prefix = Simd::from(*prefix);
                let cur_max = _mm_broadcast_ss(&cur_prefix[3]);

                *prefix = _mm_max_ps(acc, *prefix);
                acc = _mm_max_ps(acc, cur_max);
            }

            // safety: sizeof(__m256) ==  sizeof(f32) * 8, meaning it is well-aligned
            let flat_prefixes = unsafe { transmute::<&[__m256], &[f32]>(&prefix_max) };

            let _surface = zip(flat_angles, flat_prefixes)
                .map(|(&mut angle, &prefix)| angle >= prefix)
                .collect::<Vec<bool>>();

            let sum = zip(&distances, &_surface).fold(
                0.0f32,
                |surface_area, (&(distance, _), &visible)| {
                    if visible {
                        distance.mul_add(TAN_ONE_RAD, surface_area)
                    } else {
                        surface_area
                    }
                },
            );

            // safety: it is guaranteed by the rotation kernel that if the index is
            // greater than zero that it is in-bounds. This saves ~10% of bounds checks
            #[expect(
                clippy::as_conversions,
                clippy::cast_sign_loss,
                reason = "result_idx should be in [0, 2^31]"
            )]
            unsafe {
                *result.get_unchecked_mut(result_idx as usize) += sum;
            }
        }
    }
}

fn total_viewshed(elevation_map: &[i16], indexes: &[i32], max_los: usize, result: &mut [f32]) {
    if cfg!(target_feature = "avx2") {
        unsafe { total_viewshed_vector(elevation_map, indexes, max_los, result) }
    }
}

/// `kernel` is a CPU-based total viewshed kernel. It makes use of image rotation to
/// optimize the cache locality of all lookups for a total viewshed calculation
fn kernel(elevations: &[i16], max_los_points: usize, angle: usize, res: &mut [f32]) {
    assert!(angle < 360, "angle must be [0, 360)");
    let mut start = Instant::now();

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "angle is [0,360), not more than 2^54"
    )]
    let (indexes, rotated_elevations) = generate_rotation(elevations, angle as f64, max_los_points);

    tracing::info!(
        "rotated {:?} in {:?}, calculating kernel",
        angle,
        start.elapsed()
    );

    start = Instant::now();

    total_viewshed(&rotated_elevations, &indexes, max_los_points, res);
    tracing::info!("kernel for {} run in: {:?}", angle, start.elapsed());
}

/// `multithreaded_kernel` parallelizes CPU kernel calculations for a `core_count` and calculates
/// `num_angles` different angles
fn multithreaded_kernel(
    elevations: &[i16],
    max_los_points: usize,
    num_angles: usize,
    core_count: usize,
) -> Vec<f32> {
    thread::scope(|scope| {
        let threads = (0..core_count)
            .map(|start_angle: usize| {
                scope.spawn(move || {
                    let mut res = vec![0.0f32; max_los_points * max_los_points];
                    for angle in (start_angle..num_angles).step_by(core_count) {
                        kernel(elevations, max_los_points, angle, &mut res);
                    }
                    res
                })
            })
            .collect::<Vec<_>>();

        let mut res = vec![0.0f32; max_los_points * max_los_points];
        #[expect(
            clippy::unwrap_used,
            reason = "if the thread doesn't join, the program should terminate"
        )]
        for thread in threads {
            res = zip(res, thread.join().unwrap())
                .map(|(acc, heatmap)| acc + heatmap)
                .collect();
        }
        res
    })
}

/// `NUM_CORES` is the physical number of cores on a machine. Currently hardcoded to 8
/// as that is what an i9900k has, and is a common configuration.
/// TODO find a good syscall for this
const NUM_CORES: usize = 8;

impl<'compute> Compute<'compute> {
    /// Instantiate.
    pub fn new(
        backend: crate::config::Backend,
        process: Vec<crate::config::Process>,
        state_directory: Option<std::path::PathBuf>,
        maybe_output_directory: Option<std::path::PathBuf>,
        dem: &'compute mut crate::dem::DEM,
        rings_per_km: f32,
        observer_height: f32,
    ) -> Result<Self> {
        let total_bands = dem.computable_points_count * 2;

        let rings_per_band = if Self::is_process_viewsheds(&process) {
            Self::ring_count_per_band(rings_per_km, dem.max_line_of_sight)
        } else {
            1
        };
        let total_reserved_rings = if Self::is_process_viewsheds(&process) {
            usize::try_from(total_bands)? * rings_per_band
        } else {
            1
        };

        let storage = if Self::is_process_viewsheds(&process) {
            match &maybe_output_directory {
                Some(output_directory) => {
                    Some(crate::output::ring_data::Storage::new(output_directory)?)
                }
                None => None,
            }
        } else {
            None
        };

        let constants = kernel::constants::Constants {
            total_bands,
            max_los_as_points: dem.max_los_as_points,
            dem_width: dem.width,
            tvs_width: dem.tvs_width,
            observer_height,
            reserved_rings_per_band: u32::try_from(rings_per_band)?,
            process: Self::bitmask_flags_for_kernel(&process),
            ..Default::default()
        };

        let vulkan = if matches!(backend, crate::config::Backend::Vulkan) {
            let elevations = dem.elevations.clone();
            dem.elevations = Vec::new(); // Free up some RAM.
            Some(super::vulkan::Vulkan::new(
                constants,
                elevations,
                usize::try_from(dem.size)?,
                usize::try_from(dem.band_deltas_size())?,
                total_reserved_rings,
            )?)
        } else {
            None
        };

        Ok(Self {
            backend,
            process,
            vulkan,
            state_directory,
            output_directory: maybe_output_directory,
            storage,
            dem,
            constants,
            total_reserved_rings,
            total_surfaces: Vec::default(),
            ring_data: Vec::default(),
            longest_lines: Vec::default(),
        })
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Accuracy isn't needed, we're just calculating a value to help find minimum RAM usage."
    )]
    /// Calculate the expected number of rings per band of sight.
    pub const fn ring_count_per_band(rings_per_km: f32, max_line_of_sight: u32) -> usize {
        let meters_per_km = 1000.0;
        let band_length_in_km = (max_line_of_sight as f32) / meters_per_km;
        (band_length_in_km * rings_per_km) as usize
    }

    /// Are we computing everything?
    fn is_process_everything(process: &[crate::config::Process]) -> bool {
        process.contains(&crate::config::Process::All)
    }

    /// Are we computing total surface areas?
    pub fn is_process_surfaces(process: &[crate::config::Process]) -> bool {
        Self::is_process_everything(process)
            || process.contains(&crate::config::Process::TotalSurfaces)
    }

    /// Are we computing viewsheds?
    pub fn is_process_viewsheds(process: &[crate::config::Process]) -> bool {
        Self::is_process_everything(process) || process.contains(&crate::config::Process::Viewsheds)
    }

    /// Are we computing total surface areas?
    pub fn is_process_longest_lines(process: &[crate::config::Process]) -> bool {
        Self::is_process_everything(process)
            || process.contains(&crate::config::Process::LongestLines)
    }

    /// Create a GPU-friendly bitmask of flags to use in the kernel.
    pub fn bitmask_flags_for_kernel(processes: &[crate::config::Process]) -> u32 {
        use kernel::constants as kernel;
        let mut flags = 0u32;
        for process in processes {
            match process {
                crate::config::Process::All => {
                    flags |= kernel::Flag::TotalSurfaces.bit() | kernel::Flag::RingData.bit();
                }
                crate::config::Process::TotalSurfaces => {
                    flags |= kernel::Flag::TotalSurfaces.bit();
                }
                crate::config::Process::Viewsheds => flags |= kernel::Flag::RingData.bit(),
                crate::config::Process::LongestLines => {
                    flags |= kernel::Flag::LongestLines.bit();
                }
            }
        }
        flags
    }

    /// Do all computations.
    pub fn run(&mut self) -> Result<()> {
        let mut sector_surfaces = if Self::is_process_surfaces(&self.process) {
            let blank = vec![0.0; usize::try_from(self.dem.computable_points_count)?];
            self.total_surfaces.clone_from(&blank);
            blank
        } else {
            Vec::new()
        };

        if Self::is_process_viewsheds(&self.process) && self.output_directory.is_some() {
            self.save_ring_metadata()?;
        }

        let mut longest_lines = if Self::is_process_longest_lines(&self.process) {
            let blank = vec![0.0; usize::try_from(self.dem.computable_points_count)?];
            self.longest_lines.clone_from(&blank);
            blank
        } else {
            Vec::new()
        };

        if matches!(self.backend, crate::config::Backend::CPU) {
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "elevations start out as i16s, and i16 -> f32 -> i16 is lossless"
            )]
            let elevations = self
                .dem
                .elevations
                .iter()
                .map(|&x| x as i16)
                .collect::<Vec<i16>>();

            #[expect(clippy::as_conversions, reason = "max_los_as_points is ")]
            let surfaces = multithreaded_kernel(
                &elevations,
                self.dem.max_los_as_points as usize,
                360,
                NUM_CORES,
            );

            self.add_sector_surfaces_to_running_total(&surfaces);
            self.render_total_surfaces()?;
            return Ok(());
        }

        for angle in 0..crate::axes::SECTOR_STEPS {
            self.load_or_compute_cache(angle)?;
            let mut sector_ring_data = vec![0; self.total_reserved_rings];
            self.compute_sector(
                angle,
                &mut sector_surfaces,
                &mut sector_ring_data,
                &mut longest_lines,
            )?;

            if Self::is_process_viewsheds(&self.process) {
                match &self.output_directory {
                    Some(_) => {
                        self.save_sector_ring_data(angle, &sector_ring_data)?;
                    }
                    None => self.ring_data.push(sector_ring_data.clone()),
                }
            }

            if Self::is_process_surfaces(&self.process) {
                self.add_sector_surfaces_to_running_total(&sector_surfaces);
                self.render_total_surfaces()?;
            }

            if Self::is_process_longest_lines(&self.process) {
                self.increment_longest_lines(&longest_lines);
                self.render_longest_lines()?;
            }
        }

        Ok(())
    }

    /// Either load cache from the filesystem or create and save it.
    fn load_or_compute_cache(&mut self, angle: u16) -> Result<()> {
        let maybe_cache = if let Some(state_directory) = self.state_directory.clone() {
            Some(crate::cache::Cache::new(
                &state_directory,
                self.dem.width,
                angle,
            ))
        } else {
            None
        };

        if let Some(cache) = &maybe_cache {
            cache.ensure_directories_exists()?;

            if cache.is_cache_exists {
                tracing::debug!(
                    "Loading cache from: {}/*/{}",
                    cache.base_directory.display(),
                    angle
                );
                self.dem.band_deltas = cache.load_band_deltas()?;
                self.dem.band_distances = cache.load_distances()?;
                return Ok(());
            }

            tracing::warn!(
                "Cached data not found at: {}/*/{}. So computing now...",
                cache.base_directory.display(),
                angle
            );
        } else {
            tracing::warn!("Forcing computation of cache for angle {angle}°...");
        }

        self.dem.calculate_axes(f32::from(angle))?;
        self.dem.compile_band_data()?;

        if let Some(cache) = &maybe_cache {
            cache.save_band_deltas(&self.dem.band_deltas)?;
            cache.save_distances(&self.dem.band_distances)?;
        }

        Ok(())
    }

    /// Add the accumulated total surface areas for the current sector to the running total.
    fn add_sector_surfaces_to_running_total(&mut self, cumulative_surfaces: &[f32]) {
        for (left, right) in self
            .total_surfaces
            .iter_mut()
            .zip(cumulative_surfaces.iter())
        {
            *left += right;
        }
    }

    /// Check to see if this angle increases the current longest line of sight for the point.
    fn increment_longest_lines(&mut self, longest_lines: &[f32]) {
        for (left, right) in self.longest_lines.iter_mut().zip(longest_lines.iter()) {
            if right > left {
                *left = *right;
            }
        }
    }

    /// The metadata needed to reconstruct viewsheds based on the DEM and reserved rings.
    pub fn metadata(&self) -> Result<crate::output::ring_data::MetaData> {
        Ok(crate::output::ring_data::MetaData {
            width: self.dem.width,
            scale: self.dem.scale,
            max_line_of_sight: self.dem.max_line_of_sight,
            reserved_ring_size: usize::try_from(self.constants.reserved_rings_per_band)?,
            sector_shift: crate::axes::SECTOR_SHIFT,
            centre: self.dem.centre,
        })
    }

    /// Save band deltas to cache.
    pub fn save_sector_ring_data(&self, sector: u16, ring_data: &[u32]) -> Result<()> {
        let Some(storage) = self.storage.as_ref() else {
            color_eyre::eyre::bail!("Tried to save sector ring data without any active storage.");
        };

        storage.save_sector(sector, ring_data)?;
        Ok(())
    }

    /// Save the metadata for the ring data (aka viewsheds).
    pub fn save_ring_metadata(&self) -> Result<()> {
        let Some(storage) = self.storage.as_ref() else {
            color_eyre::eyre::bail!("Tried to save ring metadata without any active storage.");
        };

        storage.save_metadata(&self.metadata()?)?;
        Ok(())
    }

    /// Render a heatmap and `.bt` file of the total surface areas for each point within the computable area of the
    /// DEM.
    fn render_total_surfaces(&self) -> Result<()> {
        let Some(output_dir) = &self.output_directory else {
            return Ok(());
        };

        crate::output::png::save(
            &self.total_surfaces,
            self.dem.tvs_width,
            self.dem.tvs_width,
            output_dir.join("total_surfaces.png"),
        )?;

        crate::output::bt::save(
            self.dem,
            &self.total_surfaces,
            &output_dir.join("total_surfaces.bt"),
        )?;

        Ok(())
    }

    /// Render a heatmap and `.bt` of the longest lines of sight for each point within the computable area of the
    /// DEM.
    fn render_longest_lines(&self) -> Result<()> {
        let Some(output_dir) = &self.output_directory else {
            return Ok(());
        };

        crate::output::png::save(
            &self.longest_lines,
            self.dem.tvs_width,
            self.dem.tvs_width,
            output_dir.join("longest_lines.png"),
        )?;

        crate::output::bt::save(
            self.dem,
            &self.longest_lines,
            &output_dir.join("longest_lines.bt"),
        )?;

        Ok(())
    }

    /// Compute a single sector.
    fn compute_sector(
        &mut self,
        angle: u16,
        cumulative_surfaces: &mut [f32],
        ring_data: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        tracing::info!("Running kernel for {angle}°");
        match self.backend {
            crate::config::Backend::CPU => {
                self.compute_sector_cpu(cumulative_surfaces, ring_data, longest_lines);
            }
            crate::config::Backend::Vulkan => {
                self.compute_sector_vulkan(cumulative_surfaces, ring_data, longest_lines)?;
            }

            #[expect(clippy::unimplemented, reason = "Coming Soon!")]
            crate::config::Backend::Cuda => unimplemented!(),
        }

        Ok(())
    }

    /// Do a whole sector calculation on the GPU using Vulkan.
    fn compute_sector_vulkan(
        &mut self,
        cumulative_surfaces: &mut [f32],
        rings: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        let Some(gpu) = self.vulkan.as_mut() else {
            color_eyre::eyre::bail!("`self.gpu` not instantiated yet.");
        };

        let (surfaces_data, rings_data, longest_lines_data) =
            gpu.run(&self.dem.band_distances, &self.dem.band_deltas)?;
        if Self::is_process_surfaces(&self.process) {
            cumulative_surfaces.copy_from_slice(surfaces_data.as_slice());
        }
        if Self::is_process_viewsheds(&self.process) {
            rings.copy_from_slice(rings_data.as_slice());
        }
        if Self::is_process_longest_lines(&self.process) {
            longest_lines.copy_from_slice(longest_lines_data.as_slice());
        }
        Ok(())
    }

    /// Do a whole sector calculation on the CPU.
    fn compute_sector_cpu(
        &self,
        cumulative_surfaces: &mut [f32],
        ring_data: &mut [u32],
        longest_lines: &mut [f32],
    ) {
        for kernel_id in 0..self.constants.total_bands {
            kernel::kernel::kernel(
                kernel_id,
                &self.constants,
                &self.dem.elevations,
                &self.dem.band_distances,
                &self.dem.band_deltas,
                cumulative_surfaces,
                ring_data,
                longest_lines,
            );
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    pub fn make_dem() -> crate::dem::DEM {
        crate::dem::DEM::new(
            crate::projection::LatLonCoord((33.33, 33.33).into()),
            9,
            1.0,
            3,
        )
        .unwrap()
    }

    pub fn compute<'dem>(dem: &'dem mut crate::dem::DEM, elevations: &[i16]) -> Compute<'dem> {
        dem.elevations = elevations.iter().map(|&x| f32::from(x)).collect();
        let mut compute = Compute::new(
            crate::config::Backend::CPU,
            vec![
                crate::config::Process::TotalSurfaces,
                crate::config::Process::Viewsheds,
            ],
            None,
            None,
            dem,
            5000.0,
            1.8,
        )
        .unwrap();
        compute.run().unwrap();
        compute
    }

    fn create_viewshed(elevations: &[i16], pov_id: u32) -> Vec<String> {
        let mut dem = make_dem();
        let compute = compute(&mut dem, elevations);
        let ring_data = compute.ring_data;
        crate::output::ascii::OutputASCII::convert(
            &dem,
            pov_id,
            &ring_data,
            crate::compute::Compute::ring_count_per_band(5000.0, 3),
        )
        .unwrap()
    }

    #[rustfmt::skip]
    pub fn single_peak_dem() -> Vec<i16> {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 3, 3, 3, 3, 3, 1, 0,
            0, 1, 3, 6, 6, 6, 3, 1, 0,
            0, 1, 3, 6, 9, 6, 3, 1, 0,
            0, 1, 3, 6, 6, 6, 3, 1, 0,
            0, 1, 3, 3, 3, 3, 3, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    }

    #[rustfmt::skip]
    pub fn double_peak_dem() -> Vec<i16> {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 3, 3, 3, 3, 3, 3, 4,
            0, 1, 3, 4, 4, 4, 4, 4, 3,
            0, 1, 3, 4, 6, 4, 4, 4, 3,
            0, 1, 3, 4, 4, 4, 5, 5, 3,
            0, 1, 3, 4, 4, 5, 9, 5, 3,
            0, 1, 1, 4, 4, 5, 5, 5, 3,
            0, 0, 4, 1, 3, 3, 3, 3, 3
        ]
    }

    #[test]
    fn single_peak_totals() {
        let mut dem = make_dem();
        let compute = compute(&mut dem, &single_peak_dem());
        #[rustfmt::skip]
        assert_eq!(
            compute.total_surfaces,
            [
                2543.4292, 1649.654, 2696.0396,
                1641.8002, 3197.66, 1641.8002,
                2696.0396, 1649.6539, 2543.4292
            ]
        );
    }

    #[test]
    fn double_peak_totals() {
        let mut dem = make_dem();
        let compute = compute(&mut dem, &double_peak_dem());
        #[rustfmt::skip]
        assert_eq!(
            compute.total_surfaces,
            [
                2687.689, 2546.9956, 2622.3494,
                2564.7678, 3231.647, 2239.714,
                2604.2551, 2186.5012, 1768.3433
            ]
        );
    }

    // Here we use a simple ASCII representation of the opening and closing of ring
    // sectors:
    //   A '.' can be either inside or outisde a ring sector.
    //   A '+' is the opening and a '-' is the closing.
    //   A '±' represents both an opening and closing (from different sectors).
    //
    // Eg; Given this ASCII art profile of 2 mountain peaks, where the observer is
    // 'X':
    //
    //                     .*`*.
    //                  .*`  |  `*.
    //       .      X.*`     |     `*.
    //    .*`|`*. .*`        |        `*.
    // .*`   |   `  |        |           `*.
    //       |      |        |
    // there would be 2 ring sectors, both opening at the same point but looking
    // in different directions:
    //       |      |        |
    // ......-......+........-..............
    //
    // Or to use 0s and 1s to show the surfaces seen by the observer:
    //
    // 0000001111111111111111100000000000000
    mod viewsheds {
        #[test]
        fn summit() {
            let viewshed = super::create_viewshed(&super::single_peak_dem(), 40);
            #[rustfmt::skip]
            assert_eq!(
                viewshed,
                [
                    ". . . . . . . . .",
                    ". ± ± ± ± ± ± ± .",
                    ". ± ± ± . ± ± ± .",
                    ". ± ± . . . ± ± .",
                    ". ± . . o . . ± .",
                    ". ± ± . . . ± ± .",
                    ". ± ± ± . ± ± ± .",
                    ". ± ± ± ± ± ± ± .",
                    ". . . . . . . . ."]
            );
        }

        #[test]
        fn off_summit() {
            let viewshed = super::create_viewshed(&super::single_peak_dem(), 30);
            #[rustfmt::skip]
            assert_eq!(
                viewshed,
                [
                    "± ± ± ± ± ± ± . .",
                    "± ± ± . ± ± ± . .",
                    "± ± . . ± ± ± . .",
                    "± . . o . . ± . .",
                    "± ± ± . . ± ± . .",
                    "± ± ± . ± ± . . .",
                    "± ± ± ± ± . . . .",
                    ". . . . . . . . .",
                    ". . . . . . . . ."
                ]
            );
        }

        #[test]
        fn double_peak() {
            let viewshed = super::create_viewshed(&super::double_peak_dem(), 30);
            #[rustfmt::skip]
            assert_eq!(
                viewshed,
                [
                    "± ± ± ± ± ± ± . .",
                    "± ± ± . ± ± ± . .",
                    "± ± . . ± ± ± . .",
                    "± . . o . . ± . .",
                    "± ± ± . . ± ± . .",
                    "± ± ± . ± ± . . .",
                    "± ± ± ± ± . ± . .",
                    ". . . . . . . . .",
                    ". . . . . . . . ."
                ]
            );
        }
    }
}
