# Learning Ordinary Differential Equations with the Line Integral Loss Function

Contains a PyTorch implementation that generates the following plot:

<div align="center">
 <img src="loss.png" width="600">
 <p>Testing loss values for all models.</p>
</div>

Requires the [torchdiffeq](https://github.com/rtqichen/torchdiffeq) package to run.

## Instructions:

Run the four model training scripts separately:

* `python regression.py`
* `python node.py`
* `python sonode.py`
* `python line_integral.py`

Each of these scripts writes a txt file containing the loss history

Then run the plotting script to generate the figure above:

* `python loss_plot.py`