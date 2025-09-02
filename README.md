<div align="center">
    <h1 style="font-size: large; font-weight: bold;">Kernel-change-point-detection</h1>
</div><div align="center">
	<a href="#">
        <img src="https://img.shields.io/badge/Python-3.11%20--%203.12-blue">
    </a>
    </a>
	<a href="_static/pylint/pylint.txt">
        <img src="_static/pylint/pylint.svg" alt="Pylint Score">
    </a>
    <a href="_static/flake8/index.html">
        <img src="_static/flake8/flake8.svg" alt="Flake8 Report">
    </a>
    <a href="_static/coverage/index.html">
        <img src="_static/coverage/coverage.svg" alt="Coverage report">
    </a>
</div>
<br>


Kernel change point detection is a python library used for anomaly detection on time-series.

## Getting started

To utilize this library, ensure that you have Python 3.9 or a later version installed.

### Setting environement
```bash
pip install virtualenv
virtualenv -p <path/to/python3.10> myenv
source myenv/bin/activate
```
### Installation
Once your virtual environment is activated, you can install the Kernel-point-change-detection library directly from Pypi by typing :

```bash
pip install kcpdi
```

This command will install the kcpdi package and all required dependencies.
Within your python code, you can then access the library functions by using the following import statement:

```
import kcpdi
```

## Usage

Our algorithm takes pre-processed (interpolated to a fixed time grid, normalized, etc.) and offline (not streaming) multidimensional time series data and runs linear kernel anomaly/change-point detection on it in order to output a list of likely anomaly/change-points in time.
The method is based on results in Arlot et al. (2019).

We have started to develop automated post-hoc visualization tools in order to provide intuitive explicability in the output results in order to attribute a ranked (in terms of importance) list of individual time series for each detected anomaly/change-point. This is because it is otherwise difficult to know 'why' a change-point was detected at a certain point if there are hundreds or thousands of individual concurrent time series. The latter tool is semi-dependent of the anomaly/change-point detection step.

## Examples and demonstrators

The following [notebook](examples/Example.ipynb) incorporates the most important features:

## TADkit link

In order to allow the kernel change-point method to be integrated into a Python package based around score samples, we have implemented a function **kcp_ss** which takes the **kcp_ds** output list of change-points and turns them into scores at **all time indices**.

<p>We give a score of 1 at each detected change-point. Then, we define a decay_parameter <em>γ</em> (default <em>γ = 1</em>). Then,</p>

<ul>
  <li>For any time index that exactly corresponds to a detected change-point index, its score is 1.</li>
  
  <li>For any time index <em>t<sub>*</sub></em> before the first detected change-point time index <em>t<sub>1</sub></em>, its score is 
    <p>$$\left(\frac{1}{2}\right)^{\gamma |t_1 - t_*|}$$</p></li>
  
  <li>For any time index <em>t<sub>*</sub></em> after the last detected change-point time index <em>t<sub>last</sub></em>, its score is 
    <p>$$\left(\frac{1}{2}\right)^{\gamma |t_{last} - t_*|}$$</p></li>
  
  <li>For any time index <em>t<sub>*</sub></em> between change-point time indices <em>t<sub>j</sub></em> and <em>t<sub>j+1</sub></em>, its score is the average of the left and right scores:
    <p>$$\frac{1}{2}\left( \left(\frac{1}{2}\right)^{\gamma |t_{j} - t_*|} + \left(\frac{1}{2}\right)^{\gamma |t_{j+1} - t_*|} \right)$$</p></li>
</ul>

## Contributors and Support

Responsible : **Kevin Bleakley, Sylvain Arlot (INRIA)**

<p align="center">
  Kernel change point detection is a library developped by INRIA and supported by the  
  <a href="https://www.confiance.ai/" title="Confiance.ai">
   <img src="https://www.trustworthy-ai-foundation.eu/wp-content/uploads/2025/07/M0302_LOGO-ETAIA_RVB_2000px.png"  height="70">
  </a>
</p>