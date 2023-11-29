# -*- coding: utf-8 -*-
"""
Main script to run all services in JungleGym
======================

This is the main script to start all services required in JungleGym. This is the only script that the user should run to start the project.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 0.0.9
Status: Development
Python version: 3.9.15
"""
#External libraries:
import subprocess

def start_visualizer():
    return subprocess.Popen(["streamlit", "run", "Welcome.py", "--server.port", "8000"])

def main():
    visualizer_process = start_visualizer()

    try:
        # wait for both processes to complete
        visualizer_process.communicate()
    except KeyboardInterrupt:
        visualizer_process.terminate()

if __name__ == "__main__":
    main()
