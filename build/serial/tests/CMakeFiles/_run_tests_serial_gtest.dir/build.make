# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jethro/IWR1642_ROS/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jethro/IWR1642_ROS/build

# Utility rule file for _run_tests_serial_gtest.

# Include the progress variables for this target.
include serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/progress.make

_run_tests_serial_gtest: serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/build.make

.PHONY : _run_tests_serial_gtest

# Rule to build all files generated by this target.
serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/build: _run_tests_serial_gtest

.PHONY : serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/build

serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/clean:
	cd /home/jethro/IWR1642_ROS/build/serial/tests && $(CMAKE_COMMAND) -P CMakeFiles/_run_tests_serial_gtest.dir/cmake_clean.cmake
.PHONY : serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/clean

serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/depend:
	cd /home/jethro/IWR1642_ROS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jethro/IWR1642_ROS/src /home/jethro/IWR1642_ROS/src/serial/tests /home/jethro/IWR1642_ROS/build /home/jethro/IWR1642_ROS/build/serial/tests /home/jethro/IWR1642_ROS/build/serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : serial/tests/CMakeFiles/_run_tests_serial_gtest.dir/depend

