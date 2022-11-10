# qcpcp : quality control of precipitation observations
 Space-Time Statistical Quality Control of Extreme Precipitation Observation
 
 ##### Goal:
 
 0) Transform data using Box-Cox transormation with suitable parameter (example code 1)
 1) Find outlier in precipitation data by cross-validation approach using neighboring observations (example code 2)
 2) Identified outliers (a false observation or a single event) should be verified by discharge or radar data.
 3) Repeat the procedure over several temporal aggregations (to account for advection)
 
### Reference paper:
El Hachem, A., Seidel, J., Imbery, F., Junghänel, T., and Bárdossy, A.: Technical Note: Space-Time Statistical Quality Control of Extreme Precipitation Observations, Hydrol. Earth Syst. Sci. Discuss. [preprint], https://doi.org/10.5194/hess-2022-177, in review, 2022. 

#### Flowchart
![flowchart_outliers_2](https://user-images.githubusercontent.com/22959071/201058588-cd97bec4-693a-4c45-aefb-1a9ec62322de.png)

##### Box-Cox transformation

##### Identified outlier

##### Event spatial configuration 

#### Data availability:

The precipitation data and the radar data were made available by the German Weather Service (DWD) [https://opendata.dwd.de/climate_environment/CDC/]. The discharge data were made available by the environmental state of Bavaria and can be requested [https://www.lfu.bayern.de/index.htm]

#### Used packages for Github example code:
1) PyKrige: Benjamin Murphy, Roman Yurchak, & Sebastian Müller. (2022). GeoStat-Framework/PyKrige: v1.7.0 (v1.7.0). Zenodo. https://doi.org/10.5281/zenodo.7008206
2) adjusttext: https://adjusttext.readthedocs.io/en/latest/
3) statsmodels: https://www.statsmodels.org/devel/
