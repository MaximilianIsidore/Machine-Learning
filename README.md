# Machine-Learning using Python

# Setting Up Python Environment and Running the Program

This guide will walk you through setting up a Python environment on both Windows and Linux, installing the necessary packages, and running the provided Python programs.

## Table of Contents
- [Machine-Learning using Python](#machine-learning-using-python)
- [Setting Up Python Environment and Running the Program](#setting-up-python-environment-and-running-the-program)
  - [Table of Contents](#table-of-contents)
  - [Setting Up Python Environment](#setting-up-python-environment)
    - [Windows](#windows)
    - [Linux](#linux)
  - [Installing Required Packages](#installing-required-packages)
    - [Required Packages](#required-packages)
    - [Installation Command](#installation-command)

## Setting Up Python Environment

### Windows

1. **Install Python:**
   - Download the latest version of Python from the official website: [Python Downloads](https://www.python.org/downloads/).
   - Run the installer and make sure to check the box that says "Add Python to PATH" before clicking on "Install Now".

2. **Install pip (Python package manager):**
   - Python 3.x comes with `pip` pre-installed. You can check if it's installed by running:
     ```bash
     pip --version
     ```

3. **Create a Virtual Environment:**
   - Open Command Prompt and navigate to your project directory.
   - Create a virtual environment using:
     ```bash
     python -m venv myenv
     ```
   - Activate the virtual environment:
     ```bash
     myenv\Scripts\activate
     ```

### Linux

1. **Install Python:**
   - Most Linux distributions come with Python pre-installed. You can check the version with:
     ```bash
     python3 --version
     ```

2. **Install pip:**
   - If pip is not installed, you can install it using:
     ```bash
     sudo apt-get install python3-pip
     ```

3. **Create a Virtual Environment:**
   - Open a terminal and navigate to your project directory.
   - Create a virtual environment using:
     ```bash
     python3 -m venv myenv
     ```
   - Activate the virtual environment:
     ```bash
     source myenv/bin/activate
     ```

## Installing Required Packages

Once the virtual environment is activated, you need to install the required packages. Below is a list of the required packages and the command to install them:

### Required Packages

- `numpy`
- `pandas`
- `tkinter` (Note: `tkinter` is typically included with Python but can be installed separately if needed)
- `matplotlib`

### Installation Command

To install all the required packages, run the following command in your virtual environment:

```bash
pip install numpy pandas matplotlib
