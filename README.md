# GreenAI-extension

This project calculates the energy consumed by a given piece of code snippet or methods

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install flask. Follow this to set up a virtual environment and install flask in it. [Install Flask in a Virtual Env.](https://phoenixnap.com/kb/install-flask)

```bash
pip install Flask
```

First verify if perf is already available in you Linux system, if not then install separately:
```bash
sudo apt install linux-tools-`uname -r`
```

## Usage

Read a bit about measuring Software Energy Consumption and how it is done in Linux using the Profiling Tool known as [Perf](https://perf.wiki.kernel.org/index.php/Main_Page).

```bash
sudo perf stat -e power/energy-cores/,power/energy-ram/,power/energy-gpu/,power/energy-pkg/,power/energy-psys/ sleep 5
```
this would give the following output:

```bash
Performance counter stats for 'system wide':

              3.59 Joules power/energy-cores/                                         
              8.18 Joules power/energy-ram/                                           
              1.63 Joules power/energy-gpu/                                           
             12.74 Joules power/energy-pkg/                                           
             51.10 Joules power/energy-psys/                                          

       5.001965465 seconds time elapsed
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
