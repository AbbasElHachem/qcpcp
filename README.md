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
![Skew_before_after](https://user-images.githubusercontent.com/22959071/201070653-e2d9d788-1567-4080-8b0e-89e0cfa75d04.png)

##### Average transformation factor
![Transf_factor_lambda](https://user-images.githubusercontent.com/22959071/201071596-219dbedf-112a-431e-9c7a-0b77c8c5fed9.png)

-----------------------------------------------------------------------------------------------

#### Identified outlier [Python code 2]

##### Time series target and neighbors
![stn_P03231_ngbrs_2008_05_14 08_00_00](https://user-images.githubusercontent.com/22959071/201071744-7037b069-5fb5-46fc-a889-b58b2c8f76d2.png)

##### Event spatial configuration with Radar image
![radar_stn_P03231_2008_05_14 08_00_00__after](https://user-images.githubusercontent.com/22959071/201071903-7c54f691-4a69-454f-8c0d-1dfdf97cab7c.png)

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
