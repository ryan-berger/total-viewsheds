//! Convenience function for internal use. Dumps all the CLI usage instructions. Useful for
//! updating the README.

use color_eyre::Result;

/// main dump usage function
#[expect(clippy::print_stdout, reason = "We need to output the usage.")]
pub fn dump_full_usage_for_readme() -> Result<()> {
    use clap::CommandFactory as _;
    let mut command = crate::config::Config::command();

    let mut buffer: Vec<u8> = Vec::default();
    command.build();

    write_help(&mut buffer, &mut command, 0)?;
    println!("{}", String::from_utf8(buffer)?);

    Ok(())
}

/// Recursive function to print the usage instructions for each subcommand.
fn write_help(
    buffer: &mut impl std::io::Write,
    cmd: &mut clap::Command,
    depth: usize,
) -> Result<()> {
    if cmd.get_name() == "help" {
        return Ok(());
    }

    let mut command = cmd.get_name().to_owned();
    let indent_depth = if depth == 0 || depth == 1 { 0 } else { depth };
    let indent = " ".repeat(indent_depth * 4);
    writeln!(
        buffer,
        "\n{}* {}{}",
        indent,
        command.remove(0).to_uppercase(),
        command
    )?;

    for line in cmd.render_long_help().to_string().lines() {
        writeln!(buffer, "{indent}  {line}")?;
    }

    for sub in cmd.get_subcommands_mut() {
        writeln!(buffer)?;
        write_help(buffer, sub, depth + 1)?;
    }

    Ok(())
}
