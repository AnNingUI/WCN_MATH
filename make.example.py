#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import platform
from pathlib import Path

# ======================================
# Paths
# ======================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parent
BUILD_DIR = PROJECT_DIR / "build"
EXAMPLES_DIR = PROJECT_DIR / "examples"

# ======================================
# Defaults
# ======================================
is_no_dst = False
file_name = None
cc_compiler = "gcc"
link_mode = "static"  # static | shared

# ======================================
# Usage
# ======================================

def usage(exit_code=1):
    print()
    print(f"Usage:")
    print(f"  {SCRIPT_PATH.name} -name <file> [-cc gcc|clang] [--static|--shared]")
    print()
    print("Options:")
    print("  -name       Example source file name (required)")
    print("  -cc         Compiler to use (default: gcc)")
    print("  --no_dst    Build with NO-DST enabled")
    print("  --static    Static linking (default)")
    print("  --shared    Dynamic linking")
    print("  --help      Show this help")
    print()
    sys.exit(exit_code)

# ======================================
# getopt-style parsing
# ======================================

args = sys.argv[1:]
i = 0

while i < len(args):
    arg = args[i]

    if arg == "-name":
        i += 1
        if i >= len(args):
            print("Error: -name requires a file name")
            usage()
        file_name = args[i]
        i += 1
        continue

    if arg == "-cc":
        i += 1
        if i >= len(args):
            print("Error: -cc requires a compiler")
            usage()
        cc_compiler = args[i]
        i += 1
        continue

    if arg == "--dst":
        is_no_dst = True
        i += 1
        continue

    if arg == "--static":
        link_mode = "static"
        i += 1
        continue

    if arg == "--shared":
        link_mode = "shared"
        i += 1
        continue

    if arg == "--help":
        usage(0)

    print(f'Error: Unknown option "{arg}"')
    usage()

# ======================================
# Validation
# ======================================

if not file_name:
    print("Error: -name parameter is required")
    usage()

# ======================================
# Ensure main project is built
# ======================================

if is_no_dst:
    BUILD_DIR = PROJECT_DIR / "build-no_dst"

if not BUILD_DIR.exists():
    print("Main project not built. Building now...")
    BUILD_DIR.mkdir(parents=True)
    subprocess.run(["cmake", ".."], cwd=BUILD_DIR, check=True)
    subprocess.run(["cmake", "--build", "."], cwd=BUILD_DIR, check=True)
else:
    print("Main project already built.")

# ======================================
# Resolve example
# ======================================

example_source = EXAMPLES_DIR / file_name

if not example_source.exists():
    print(f"Error: Example file not found: {example_source}")
    sys.exit(1)

output_dir = BUILD_DIR / "examples"
output_dir.mkdir(exist_ok=True)

output_name = example_source.stem
output_exe = output_dir / f"{output_name}.exe"

# ======================================
# Detect language
# ======================================

is_cpp = example_source.suffix.lower() in {".cpp", ".cxx", ".cc"}
std_flag = "-std=c++17" if is_cpp else "-std=c11"

# ======================================
# Resolve library (核心逻辑)
# ======================================

system = platform.system().lower()

candidates = []

if system == "windows":
    if link_mode == "static":
        candidates = [
            BUILD_DIR / "wcn_math.lib",
            BUILD_DIR / "libwcn_math.a",
        ]
    else:
        candidates = [
            BUILD_DIR / "wcn_math.lib",      # import lib
        ]
else:
    if link_mode == "static":
        candidates = [
            BUILD_DIR / "libwcn_math.a",
        ]
    else:
        candidates = [
            BUILD_DIR / "libwcn_math.so",
            BUILD_DIR / "libwcn_math.dylib",
        ]

lib_path = next((p for p in candidates if p.exists()), None)

if not lib_path:
    print("Error: No suitable wcn_math library found.")
    print("Searched:")
    for p in candidates:
        print(f"  {p}")
    sys.exit(1)

# ======================================
# Compile & link
# ======================================

print()
print(f'Compiling "{file_name}" with "{cc_compiler}" ({link_mode})...')
print(f"Using library: {lib_path.name}")

cmd = [
    cc_compiler,
    std_flag,
    f"-I{PROJECT_DIR / 'include'}",
    f"-I{PROJECT_DIR / 'src'}",
    str(example_source),
    str(lib_path),
    "-o",
    str(output_exe),
]

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError:
    print()
    print("Build failed.")
    sys.exit(1)

print()
print("Compiled successfully:")
print(f"  {output_exe}")
print()
