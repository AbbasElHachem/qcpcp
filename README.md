# qcpcp (qcp<sup>2</sup>): quality control of precipitation observations
 ### Space-Time Statistical Quality Control of Extreme Precipitation Observation
 
 #### Goal:
 
 0) Transform data using Box-Cox transormation with suitable parameter (example code 1)
 1) Find outlier in precipitation data by cross-validation approach using neighboring observations (example code 2)
 2) Identified outliers (a false observation or a single event) should be verified by discharge or radar data.
 3) Repeat the procedure over several temporal aggregations (to account for advection)
 
-----------------------------------------------------------------------------------------------

### Reference paper:
El Hachem, A., Seidel, J., Imbery, F., Junghänel, T., and Bárdossy, A.: Technical Note: Space-Time Statistical Quality Control of Extreme Precipitation Observations, Hydrol. Earth Syst. Sci. Discuss. [preprint], https://doi.org/10.5194/hess-2022-177, in review, 2022. 

-----------------------------------------------------------------------------------------------

#### Flowchart - Procedure
![flowchart_outliers_2](https://user-images.githubusercontent.com/22959071/201058588-cd97bec4-693a-4c45-aefb-1a9ec62322de.png)

-----------------------------------------------------------------------------------------------

## Example data and codes on Github
-----------------------------------------------------------------------------------------------

#### Study area
<img width="448" alt="Location_case_study" src="https://user-images.githubusercontent.com/22959071/201070366-ad23af3d-d7e0-42b4-a2d0-844f44e83600.png">

-----------------------------------------------------------------------------------------------

#### Box-Cox transformation [Python code 1]

##### Skewness before and after transformation
![Skew_before_after](https://user-images.githubusercontent.com/22959071/201102430-b36586ee-93a7-4058-b00c-66a7dd27bdf3.png)

##### Average transformation factor
![Transf_factor_lambda](https://user-images.githubusercontent.com/22959071/201102531-2dff63a7-ddac-4380-ba77-c81e7aedc68c.png)

-----------------------------------------------------------------------------------------------

#### Identified outlier [Python code 2]

##### Time series target and neighbors
![stn_P03231_ngbrs_2008_05_14 08_00_00](https://user-images.githubusercontent.com/22959071/201103671-cbf77a69-f137-4f24-9c97-1c3d2a4d7c2c.png)

##### Event spatial configuration with Radar image
![radar_stn_P03231_2008_05_14 08_00_00__after](https://user-images.githubusercontent.com/22959071/201103852-94f375c0-01f6-45e7-aa1f-4d28cb5b5075.png)

-----------------------------------------------------------------------------------------------

#### Data availability:


The precipitation data and the radar data were made available by the German Weather Service (DWD) [https://opendata.dwd.de/climate_environment/CDC/]. The discharge data were made available by the environmental state of Bavaria and can be requested [https://www.lfu.bayern.de/index.htm]

-----------------------------------------------------------------------------------------------

#### Used packages for Github example code:
1) PyKrige: Benjamin Murphy, Roman Yurchak, & Sebastian Müller. (2022). GeoStat-Framework/PyKrige: v1.7.0 (v1.7.0). Zenodo. https://doi.org/10.5281/zenodo.7008206
2) adjusttext: https://adjusttext.readthedocs.io/en/latest/
3) statsmodels: https://www.statsmodels.org/devel/

-----------------------------------------------------------------------------------------------

#### Note: this is an exmaple case, in the paper a modified code was used with Variogram estimation and personal kriging code. These are not updated to keep things simple.
