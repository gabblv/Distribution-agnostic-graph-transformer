**Summary**: 

Distribution-agnostic graph transformer architecture.

Specifically, the project aimed at predicting landslide extents (regression) on a highly right-skewed dataset.

It uses a non-standard loss function, the Squared Error Relevance Area (SERA), and the Mean Squared Error (MSE) as performance metric.


**Reproducibility**:

1 - Install the python environment from `pythonenv/environment.yml` ($^*$)

($^*$) If `ERROR: No matching distribution found for torch==1.13.1+cu116` is returned, please:
- Remove `torch==1.13.1+cu116` from the *yml* file
- Install the environment
- Activate the environment
- Download and install the *torch* module manually from the following link with `pip install *.whl`:

https://download.pytorch.org/whl/torch/

<img src="pythonenv/wheel_screenshot.png" alt="alt text" width="400" />
	
2 - Run `run.py`

3 - (*Optional*) Run `pred_visualization.ipynb` to visualize the predictions. Before running, please extract the *gpkg* contained in`data/custom/Wenchuan_data_final.zip` into `data/custom/`

**Note**:

`data_prep.ipynb` is also provided as reference, sharing the preparation of the original dataset as model input.
