# Filter immuno phenotypes by signifance threshold. 
import pandas as pd
import math

from Controllers.DataScienceManager import DataScienceManager as dsm

use_full_dataset = False
d_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

ols_results = d_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='model_result_ols'
)

cut_off = 1.5480817e-06
print(-1*math.log10(cut_off))

subset = ols_results[ols_results['p_1'] < cut_off].copy()
print(subset)