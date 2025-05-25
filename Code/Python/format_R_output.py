#%%
import pandas as pd
import numpy as np

dict_translate_term_names = {"factor(cpap_treated)1": r"\gls{cpap} treated (=1)",
                             "rcs(time_to_cpap_start)time_to_cpap_start": r"Time until \gls{cpap} start (linear)",
                             "rcs(time_to_cpap_start)time_to_cpap_start'": r"Time until \gls{cpap} start (non-linear, 1)",
                             "rcs(time_to_cpap_start)time_to_cpap_start''": r"Time until \gls{cpap} start (non-linear, 2)",
                             "rcs(time_to_cpap_start)time_to_cpap_start'''": r"Time until \gls{cpap} start (non-linear, 3)",
                             "rcs(time_to_cpap_end)time_to_cpap_end": r"Time until \gls{cpap} end (linear)",
                             "rcs(time_to_cpap_end)time_to_cpap_end'": r"Time until \gls{cpap} end (non-linear, 1)",
                             "rcs(time_to_cpap_end)time_to_cpap_end''": r"Time until \gls{cpap} end (non-linear, 2)",
                             "rcs(time_to_cpap_end)time_to_cpap_end'''": r"Time until \gls{cpap} end (non-linear, 3)",
                             "rcs(alder)alder": r"Age (linear)",
                             "rcs(alder)alder'": r"Age (non-linear, 1)",
                             "rcs(alder)alder''": r"Age (non-linear, 2)",
                             "rcs(alder)alder'''": r"Age (non-linear, 3)",
                             "factor(sex)1": r"Gender (=female)",
                             "rcs(bmi)bmi": r"\gls{bmi} (linear)",
                             "rcs(bmi)bmi'": r"\gls{bmi} (non-linear, 1)",
                             "rcs(bmi)bmi''": r"\gls{bmi} (non-linear, 2)",
                             "rcs(bmi)bmi'''": r"\gls{bmi} (non-linear, 3)",
                             "rcs(hba1c)hba1c": r"\gls{hba1c} (linear)",
                             "rcs(hba1c)hba1c'": r"\gls{hba1c} (non-linear, 1)",
                             "rcs(hba1c)hba1c''": r"\gls{hba1c} (non-linear, 2)",
                             "rcs(hba1c)hba1c'''": r"\gls{hba1c} (non-linear, 3)",
                             "factor(hba1c_high)1": r"\gls{hba1c} (=high)",
                             "rcs(GFR)GFR": r"\gls{gfr} (linear)",
                             "rcs(GFR)GFR'": r"\gls{gfr} (non-linear, 1)",
                             "rcs(GFR)GFR''": r"\gls{gfr} (non-linear, 2)",
                             "rcs(GFR)GFR'''": r"\gls{gfr} (non-linear, 3)",
                             "factor(GFR_high)1": r"\gls{gfr} (=high)",
                             "factor(rokare)1": r"Smoking status (=1)",
                             "rcs(kolesterol)kolesterol": r"Total cholesterol (linear)",
                             "rcs(kolesterol)kolesterol'": r"Total cholesterol (non-linear, 1)",
                             "rcs(kolesterol)kolesterol''": r"Total cholesterol (non-linear, 2)",
                             "rcs(kolesterol)kolesterol'''": r"Total cholesterol (non-linear, 3)",
                             "factor(kolesterol_high)1": r"Total cholesterol (=high)",
                             "rcs(systoliskt)systoliskt": r"\gls{sbp} (linear)",
                             "rcs(systoliskt)systoliskt'": r"\gls{sbp} (non-linear, 1)",
                             "rcs(systoliskt)systoliskt''": r"\gls{sbp} (non-linear, 2)",
                             "rcs(systoliskt)systoliskt'''": r"\gls{sbp} (non-linear, 3)",
                             "factor(systoliskt_high)1": r"\gls{sbp} (=high)",
                             "factor(stroke_history)1": "Stroke history (=1)",
                             "factor(ami_history)1": r"\gls{ami} history (=1)",
                             "factor(Civil)1": r"Widowed status (=1)",
                             "factor(HushallsTyp_RTB)2": r"Household type (=living together)",
                             "factor(HushallsTyp_RTB)3": r"Household type (=other)",
                             "factor(Sun2000niva_old)2": r"Education level (=2)",
                             "factor(Sun2000niva_old)3": r"Education level (=3)",
                             "factor(Sun2000niva_old)4": r"Education level (=4)",
                             "factor(Sun2000niva_old)5": r"Education level (=5)",
                             "factor(Sun2000niva_old)6": r"Education level (=6)",
                             "factor(Sun2000niva_old)7": r"Education level (=7)",
                             "rcs(DispInkKEHB04)DispInkKEHB04": r"Household disposable income (linear)",
                             "rcs(DispInkKEHB04)DispInkKEHB04'": r"Household disposable income (non-linear, 1)",
                             "rcs(DispInkKEHB04)DispInkKEHB04''": r"Household disposable income (non-linear, 2)",
                             "rcs(DispInkKEHB04)DispInkKEHB04'''": r"Household disposable income (non-linear, 3)",
                             "rcs(Raks_AndelArblosInk)Raks_AndelArblosInk": r"Income prop. unemployment (linear)",
                             "rcs(Raks_AndelArblosInk)Raks_AndelArblosInk'": r"Income prop. unemployment (non-linear, 1)",
                             "rcs(Raks_AndelArblosInk)Raks_AndelArblosInk''": r"Income prop. unemployment (non-linear, 2)",
                             "rcs(Raks_AndelArblosInk)Raks_AndelArblosInk'''": r"Income prop. unemployment (non-linear, 3)",
                             "rcs(Raks_AndelSjukInk)Raks_AndelSjukInk": r"Income prop. sickness (linear)",
                             "rcs(Raks_AndelSjukInk)Raks_AndelSjukInk'": r"Income prop. sickness (non-linear, 1)",
                             "rcs(Raks_AndelSjukInk)Raks_AndelSjukInk''": r"Income prop. sickness (non-linear, 2)",
                             "rcs(Raks_AndelSjukInk)Raks_AndelSjukInk'''": r"Income prop. sickness (non-linear, 3)",
                             "rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk": r"Income prop. financial aid (linear)",
                             "rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk'": r"Income prop. financial aid (non-linear, 1)",
                             "rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk''": r"Income prop. financial aid (non-linear, 2)",
                             "rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk'''": r"Income prop. financial aid (non-linear, 3)",
                             "factor(FodelseLand_EU27_2020)1": "Born in Sweden (=1)",
                             "factor(antithrombotic_agents)1": r"Antithrombotic drug usage (=1)",
                             "factor(antihypertensive_comb)1": r"Antihypertensive drug usage (=1)",
                             "factor(lipid_modifying_agents)1": r"Lipid-modifying drug usage (=1)",
                             "rcs(IV_AHI)IV_AHI": r"\gls{ahi} (linear)",
                             "rcs(IV_AHI)IV_AHI'": r"\gls{ahi} (non-linear, 1)",
                             "rcs(IV_AHI)IV_AHI''": r"\gls{ahi} (non-linear, 2)",
                             "rcs(IV_AHI)IV_AHI'''": r"\gls{ahi} (non-linear, 3)",
                             "rcs(IV_AverageSaturation)IV_AverageSaturation": "Average saturation (linear)",
                             "rcs(IV_AverageSaturation)IV_AverageSaturation'": "Average saturation (non-linear, 1)",
                             "rcs(IV_AverageSaturation)IV_AverageSaturation''": "Average saturation (non-linear, 2)",
                             "rcs(IV_AverageSaturation)IV_AverageSaturation'''": "Average saturation (non-linear, 3)",
                             "rcs(IV_ODI)IV_ODI": r"\gls{odi} (linear)",
                             "rcs(IV_ODI)IV_ODI'": r"\gls{odi} (non-linear, 1)",
                             "rcs(IV_ODI)IV_ODI''": r"\gls{odi} (non-linear, 2)",
                             "rcs(IV_ODI)IV_ODI'''": r"\gls{odi} (non-linear, 3)",
                             "rcs(alder)alder:rcs(GFR)GFR": r"Age (linear):\gls{gfr} (linear)",
                             "rcs(alder)alder':rcs(GFR)GFR": r"Age (non-linear, 1):\gls{gfr} (linear)",
                             "rcs(alder)alder'':rcs(GFR)GFR": r"Age (non-linear, 2):\gls{gfr} (linear)",
                             "rcs(alder)alder''':rcs(GFR)GFR": r"Age (non-linear, 3):\gls{gfr} (linear)",
                             "rcs(alder)alder:rcs(GFR)GFR'": r"Age (linear):\gls{gfr} (non-linear, 1)",
                             "rcs(alder)alder':rcs(GFR)GFR'": r"Age (non-linear, 1):\gls{gfr} (non-linear, 1)",
                             "rcs(alder)alder'':rcs(GFR)GFR'": r"Age (non-linear, 2):\gls{gfr} (non-linear, 1)",
                             "rcs(alder)alder''':rcs(GFR)GFR'": r"Age (non-linear, 3):\gls{gfr} (non-linear, 1)",
                             "rcs(alder)alder:rcs(GFR)GFR''": r"Age (linear):\gls{gfr} (non-linear, 2)",
                             "rcs(alder)alder':rcs(GFR)GFR''": r"Age (non-linear, 1):\gls{gfr} (non-linear, 2)",
                             "rcs(alder)alder'':rcs(GFR)GFR''": r"Age (non-linear, 2):\gls{gfr} (non-linear, 2)",
                             "rcs(alder)alder''':rcs(GFR)GFR''": r"Age (non-linear, 3):\gls{gfr} (non-linear, 2)",
                             "rcs(alder)alder:rcs(GFR)GFR'''": r"Age (linear):\gls{gfr} (non-linear, 3)",
                             "rcs(alder)alder':rcs(GFR)GFR'''": r"Age (non-linear, 1):\gls{gfr} (non-linear, 3)",
                             "rcs(alder)alder'':rcs(GFR)GFR'''": r"Age (non-linear, 2):\gls{gfr} (non-linear, 3)",
                             "rcs(alder)alder''':rcs(GFR)GFR'''": r"Age (non-linear, 3):\gls{gfr} (non-linear, 3)",
                             "rcs(alder)alder:factor(GFR_high)1": r"Age (linear):\gls{gfr} (=high)",
                             "rcs(alder)alder':factor(GFR_high)1": r"Age (non-linear, 1):\gls{gfr} (=high)",
                             "rcs(alder)alder'':factor(GFR_high)1": r"Age (non-linear, 2):\gls{gfr} (=high)",
                             "rcs(alder)alder''':factor(GFR_high)1": r"Age (non-linear, 3):\gls{gfr} (=high)",
                             "factor(cpap_treated)1:rcs(GFR)GFR": r"\gls{gfr} (linear):\gls{cpap} treated (=1)",
                             "factor(cpap_treated)1:rcs(GFR)GFR'": r"\gls{gfr} (non-linear, 1):\gls{cpap} treated (=1)",
                             "factor(cpap_treated)1:rcs(GFR)GFR''": r"\gls{gfr} (non-linear, 2):\gls{cpap} treated (=1)",
                             "factor(cpap_treated)1:rcs(GFR)GFR'''": r"\gls{gfr} (non-linear, 3):\gls{cpap} treated (=1)",
                             "factor(cpap_treated)1:factor(GFR_high)1": r"\gls{gfr} (=high):\gls{cpap} treated (=1)",
                             "rcs(alder)alder:rcs(bmi)bmi": r"Age (linear):\gls{bmi} (linear)",
                             "rcs(alder)alder':rcs(bmi)bmi": r"Age (non-linear, 1):\gls{bmi} (linear)",
                             "rcs(alder)alder'':rcs(bmi)bmi": r"Age (non-linear, 2):\gls{bmi} (linear)",
                             "rcs(alder)alder''':rcs(bmi)bmi": r"Age (non-linear, 3):\gls{bmi} (linear)",
                             "rcs(alder)alder:rcs(bmi)bmi'": r"Age (linear):\gls{bmi} (non-linear, 1)",
                             "rcs(alder)alder':rcs(bmi)bmi'": r"Age (non-linear, 1):\gls{bmi} (non-linear, 1)",
                             "rcs(alder)alder'':rcs(bmi)bmi'": r"Age (non-linear, 2):\gls{bmi} (non-linear, 1)",
                             "rcs(alder)alder''':rcs(bmi)bmi'": r"Age (non-linear, 3):\gls{bmi} (non-linear, 1)",
                             "rcs(alder)alder:rcs(bmi)bmi''": r"Age (linear):\gls{bmi} (non-linear, 2)",
                             "rcs(alder)alder':rcs(bmi)bmi''": r"Age (non-linear, 1):\gls{bmi} (non-linear, 2)",
                             "rcs(alder)alder'':rcs(bmi)bmi''": r"Age (non-linear, 2):\gls{bmi} (non-linear, 2)",
                             "rcs(alder)alder''':rcs(bmi)bmi''": r"Age (non-linear, 3):\gls{bmi} (non-linear, 2)",
                             "rcs(alder)alder:rcs(bmi)bmi'''": r"Age (linear):\gls{bmi} (non-linear, 3)",
                             "rcs(alder)alder':rcs(bmi)bmi'''": r"Age (non-linear, 1):\gls{bmi} (non-linear, 3)",
                             "rcs(alder)alder'':rcs(bmi)bmi'''": r"Age (non-linear, 2):\gls{bmi} (non-linear, 3)",
                             "rcs(alder)alder''':rcs(bmi)bmi'''": r"Age (non-linear, 3):\gls{bmi} (non-linear, 3)",
                             "factor(cpap_treated)1:factor(lipid_modifying_agents)1": r"\gls{cpap} treated (=1):lipid-modifying (=1)",
                             "factor(cpap_treated)1:factor(antithrombotic_agents)1": r"\gls{cpap} treated (=1):antithrombotic (=1)",
                             "factor(cpap_treated)1:factor(antihypertensive_comb)1": r"\gls{cpap} treated (=1):antihypertensive (=1)",
                             }


def format_columns(output):
    list_output = output.strip().splitlines()
    list_output = [line.replace("## ", "").split() for line in list_output][1:] # Remove leading hashes and column names line
    dict_output = {dict_translate_term_names[list_line[0]]: list_line[1:] for list_line in list_output if list_line[0] in dict_translate_term_names.keys()} # Translate term names
    dict_output = {k: [s for s in v if ("*" not in s) and (s != ".")] for k, v in dict_output.items()} # Remove significance level from columns
    dict_output = {k: v[:-2] + [v[-2] + v[-1]] if (len(v) >= 2) and (v[-2] == "<") else v for k, v in dict_output.items()}

    # Check if any terms were skipped
    n_skipped = len(list_output) - len(dict_output.keys())
    if n_skipped > 0:
        [print(list_line[0]) for list_line in list_output if list_line[0] not in dict_translate_term_names.keys()]

    return dict_output


def parse_p_value(p_str):
    if p_str.startswith("<"):
        return float(p_str[1:])
    else:
        return float(p_str)
    

def mark_term(row):
    try:
        if parse_p_value(row[r"$\bm{p}$"]) < 0.05:
            return r"\rowcolor{lightgray} " + row[r"\bfseries{Term}"]
        else:
            return row[r"\bfseries{Term}"]
    except ValueError:
        pass # Skip invalid strings
    return row[r"\bfseries{Term}"]


def format_table_surv(list_output):
    list_dict_outputs = []
    for i, output in enumerate(list_output):
        list_dict_outputs.append(format_columns(output))

        if i > 0:
            if list_dict_outputs[i].keys() != list_dict_outputs[i - 1].keys():
                raise Exception("Keys don't overlap! Sure the outputs are from the same model?")
    
    dict_output_comb = {k: [] for k in list_dict_outputs[0].keys()}
    for dict_output in list_dict_outputs:
        for k in list_dict_outputs[0].keys():
            dict_output_comb[k].extend(dict_output.get(k, []))

    df_table = pd.DataFrame.from_dict(dict_output_comb, orient="index", columns=[r"$\bm{\hat{\beta}}$", r"\bfseries{$\bm{e^{\hat{\beta}}}$/\gls{hr}}", r"$\bm{SE(\hat{\beta})}$", "z", r"$\bm{p}$"]).reset_index(names=r"\bfseries{Term}").drop("z", axis=1)
    df_table[r"\bfseries{Term}"] = df_table.apply(mark_term, axis=1)

    return df_table.to_latex(index=False)
    

def format_table_dac(list_output):
    list_dict_outputs = []
    for i, output in enumerate(list_output):
        list_dict_outputs.append(format_columns(output))

        if i > 0:
            if list_dict_outputs[i].keys() != list_dict_outputs[i - 1].keys():
                raise Exception("Keys don't overlap! Sure the outputs are from the same model?")
    
    dict_output_comb = {k: [] for k in list_dict_outputs[0].keys()}
    for dict_output in list_dict_outputs:
        for k in list_dict_outputs[0].keys():
            dict_output_comb[k].extend(dict_output.get(k, []))

    df_table = pd.DataFrame.from_dict(dict_output_comb, orient="index", columns=[r"$\bm{\hat{\beta}}$", r"$\bm{SE(\hat{\beta})}$", "z", r"$\bm{p}$"]).reset_index(names=r"\bfseries{Term}").drop("z", axis=1)
    df_table[r"\bfseries{$\bm{e^{\hat{\beta}}}$/\gls{hr}}"] = df_table[r"$\bm{\hat{\beta}}$"].apply(lambda beta: f"{np.exp(float(beta)):.2e}")
    df_table = df_table[[r"\bfseries{Term}", r"$\bm{\hat{\beta}}$", r"\bfseries{$\bm{e^{\hat{\beta}}}$/\gls{hr}}", r"$\bm{SE(\hat{\beta})}$", r"$\bm{p}$"]]

    df_table = df_table.replace("NaN", "NA")

    df_table[r"\bfseries{Term}"] = df_table.apply(mark_term, axis=1)

    return df_table.to_latex(index=False)


#%%
list_output = [
"""
##                                                            coef  exp(coef)
## factor(cpap_treated)1                                 8.726e-02  1.091e+00
## rcs(alder)alder                                       4.654e-01  1.593e+00
## rcs(alder)alder'                                      9.474e-02  1.099e+00
## rcs(alder)alder''                                     7.887e-01  2.201e+00
## rcs(alder)alder'''                                   -4.639e+00  9.669e-03
## factor(stroke_history)1                              -1.673e+01  5.444e-08
## factor(antithrombotic_agents)1                        5.667e-01  1.763e+00
## rcs(DispInkKEHB04)DispInkKEHB04                      -2.366e-01  7.893e-01
## rcs(DispInkKEHB04)DispInkKEHB04'                      4.666e+00  1.063e+02
## rcs(DispInkKEHB04)DispInkKEHB04''                    -2.851e+01  4.151e-13
## rcs(DispInkKEHB04)DispInkKEHB04'''                    4.705e+01  2.720e+20
## factor(hba1c_high)1                                   3.252e-01  1.384e+00
## factor(systoliskt_high)1                              2.793e-01  1.322e+00
## factor(rokare)1                                       2.469e-01  1.280e+00
## rcs(IV_ODI)IV_ODI                                     1.557e-02  1.016e+00
## rcs(IV_ODI)IV_ODI'                                    5.562e+00  2.605e+02
## rcs(IV_ODI)IV_ODI''                                  -1.326e+01  1.741e-06
## rcs(IV_ODI)IV_ODI'''                                  9.708e+00  1.644e+04
## factor(cpap_treated)1:factor(antithrombotic_agents)1 -3.953e-01  6.735e-01
""",
"""
##                                                        se(coef)      z Pr(>|z|)
## factor(cpap_treated)1                                 1.490e-01  0.585  0.55826
## rcs(alder)alder                                       3.313e-01  1.405  0.16013
## rcs(alder)alder'                                      1.183e+00  0.080  0.93617
## rcs(alder)alder''                                     6.162e+00  0.128  0.89815
## rcs(alder)alder'''                                    1.061e+01 -0.437  0.66210
## factor(stroke_history)1                               9.670e+02 -0.017  0.98620
## factor(antithrombotic_agents)1                        1.161e-01  4.882 1.05e-06
## rcs(DispInkKEHB04)DispInkKEHB04                       4.312e-01 -0.549  0.58321
## rcs(DispInkKEHB04)DispInkKEHB04'                      6.378e+00  0.732  0.46446
## rcs(DispInkKEHB04)DispInkKEHB04''                     2.263e+01 -1.260  0.20773
## rcs(DispInkKEHB04)DispInkKEHB04'''                    2.651e+01  1.775  0.07586
## factor(hba1c_high)1                                   1.064e-01  3.055  0.00225
## factor(systoliskt_high)1                              1.075e-01  2.598  0.00938
## factor(rokare)1                                       1.361e-01  1.814  0.06967
## rcs(IV_ODI)IV_ODI                                     6.012e-01  0.026  0.97933
## rcs(IV_ODI)IV_ODI'                                    9.795e+00  0.568  0.57010
## rcs(IV_ODI)IV_ODI''                                   2.194e+01 -0.605  0.54550
## rcs(IV_ODI)IV_ODI'''                                  1.621e+01  0.599  0.54930
## factor(cpap_treated)1:factor(antithrombotic_agents)1  2.202e-01 -1.795  0.07258
""",
"""                                
##                                                         
## factor(cpap_treated)1                                   
## rcs(alder)alder                                         
## rcs(alder)alder'                                        
## rcs(alder)alder''                                       
## rcs(alder)alder'''                                      
## factor(stroke_history)1                                 
## factor(antithrombotic_agents)1                       ***
## rcs(DispInkKEHB04)DispInkKEHB04                         
## rcs(DispInkKEHB04)DispInkKEHB04'                        
## rcs(DispInkKEHB04)DispInkKEHB04''                       
## rcs(DispInkKEHB04)DispInkKEHB04'''                   .  
## factor(hba1c_high)1                                  ** 
## factor(systoliskt_high)1                             ** 
## factor(rokare)1                                      .  
## rcs(IV_ODI)IV_ODI                                       
## rcs(IV_ODI)IV_ODI'                                      
## rcs(IV_ODI)IV_ODI''                                     
## rcs(IV_ODI)IV_ODI'''                                    
## factor(cpap_treated)1:factor(antithrombotic_agents)1 .  
"""
]

print(format_table_surv(list_output))

#%%
list_output = [
"""
##                                                       Penalized Est  Std. Error
## factor(cpap_treated)1                                    -0.3389997   0.1124921
## rcs(alder)alder                                           0.5224268   0.1895416
## rcs(alder)alder'                                          0.0496372   0.5730452
## rcs(alder)alder''                                        -0.6758112   2.7496410
## rcs(alder)alder'''                                        5.7206795   4.4729122
## factor(sex)1                                             -0.3420816   0.0056457
## rcs(bmi)bmi                                              -0.7057801   0.1943251
## rcs(bmi)bmi'                                              0.8272827   1.7877432
## rcs(bmi)bmi''                                             1.1262705   8.0816413
## rcs(bmi)bmi'''                                           -4.8096422  10.0915362
## factor(rokare)1                                           0.4625013   0.0073350
## factor(ami_history)1                                      0.1975812   0.0085610
## factor(stroke_history)1                                   0.3505667   0.0092775
## factor(Civil)1                                           -0.1340552   0.0076394
## factor(HushallsTyp_RTB)2                                 -0.2671695   0.0065964
## factor(HushallsTyp_RTB)3                                 -0.0626617   0.0101121
## factor(Sun2000niva_old)2                                  0.0307543   0.0095207
## factor(Sun2000niva_old)3                                 -0.0434487   0.0064233
## factor(Sun2000niva_old)4                                 -0.0814475   0.0093023
## factor(Sun2000niva_old)5                                 -0.1343848   0.0109219
## factor(Sun2000niva_old)6                                 -0.1812515   0.0108692
## factor(Sun2000niva_old)7                                 -0.2752011   0.0368742
## rcs(DispInkKEHB04)DispInkKEHB04                           0.0221395   0.0434669
## rcs(DispInkKEHB04)DispInkKEHB04'                         -3.7236679   1.0976213
## rcs(DispInkKEHB04)DispInkKEHB04''                        -1.0831016   3.5767877
## rcs(DispInkKEHB04)DispInkKEHB04'''                       14.9031724   3.5952479
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk               0.5297725   0.1010855
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk'             -6.7819212  48.4718188
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk''             0.1709860  52.6062421
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk'''           12.0327359   4.7680567
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk                   1.2615548   0.1564742
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk'                -50.0614093 239.3040836
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk''                 7.1384348 276.7581145
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk'''               56.4273897  38.4641589
## rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk                 0.4300701   0.0693960
## rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk'               -6.4020463  13.1088259
## rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk''               5.7578502  13.7412120
## factor(FodelseLand_EU27_2020)1                            0.1040616   0.0074921
## factor(antithrombotic_agents)1                            0.2343311   0.0057425
## factor(antihypertensive_comb)1                           -0.0244667   0.0057209
## factor(lipid_modifying_agents)1                          -0.1720458   0.0055470
## factor(hba1c_high)1                                       0.0953773   0.0053489
## factor(GFR_high)1                                        -0.6532521   0.0993170
## factor(kolesterol_high)1                                 -0.0471203   0.0061301
## factor(systoliskt_high)1                                 -0.2798710   0.0052580
## factor(cpap_treated)1:factor(antithrombotic_agents)1      0.0769248   0.0902398
## factor(cpap_treated)1:factor(antihypertensive_comb)1      0.0375517   0.0987867
## factor(cpap_treated)1:factor(lipid_modifying_agents)1     0.1148048   0.0917067
## rcs(alder)alder:rcs(bmi)bmi                               0.0202391   0.1512848
## rcs(alder)alder':rcs(bmi)bmi                              0.0918857   0.4785300
## rcs(alder)alder'':rcs(bmi)bmi                            -1.9522211   2.3535700
## rcs(alder)alder''':rcs(bmi)bmi                            7.0908453   3.9006839
## rcs(alder)alder:rcs(bmi)bmi'                             -0.5533291   1.4229908
## rcs(alder)alder':rcs(bmi)bmi'                             1.7669719   4.3962129
## rcs(alder)alder'':rcs(bmi)bmi'                            0.8350339  21.4238092
## rcs(alder)alder''':rcs(bmi)bmi'                         -10.7510293  35.4397906
## rcs(alder)alder:rcs(bmi)bmi''                             0.0741843   6.4343853
## rcs(alder)alder':rcs(bmi)bmi''                           -2.3081875  19.9042992
## rcs(alder)alder'':rcs(bmi)bmi''                          -5.4654472  97.2556520
## rcs(alder)alder''':rcs(bmi)bmi''                        -44.5353602 161.6735563
## rcs(alder)alder:rcs(bmi)bmi'''                            2.6189072   8.0146486
## rcs(alder)alder':rcs(bmi)bmi'''                          -4.4393853  24.9084070
## rcs(alder)alder'':rcs(bmi)bmi'''                         14.5168190 122.2803205
## rcs(alder)alder''':rcs(bmi)bmi'''                       141.3624555 204.6209293
## rcs(alder)alder:factor(GFR_high)1                         0.4869240   0.0844500
## rcs(alder)alder':factor(GFR_high)1                       -0.0250989   0.2388273
## rcs(alder)alder'':factor(GFR_high)1                      -0.3742250   1.0959204
## rcs(alder)alder''':factor(GFR_high)1                     -0.0590301   1.7130944
## factor(cpap_treated)1:factor(GFR_high)1                  -0.0896735   0.0863156
""",
"""
##                                                        z value  Pr(>|z|)    
## factor(cpap_treated)1                                  -3.0135 0.0025822 ** 
## rcs(alder)alder                                         2.7563 0.0058466 ** 
## rcs(alder)alder'                                        0.0866 0.9309735    
## rcs(alder)alder''                                      -0.2458 0.8058513    
## rcs(alder)alder'''                                      1.2790 0.2009108    
## factor(sex)1                                          -60.5919 < 2.2e-16 ***
## rcs(bmi)bmi                                            -3.6320 0.0002813 ***
## rcs(bmi)bmi'                                            0.4628 0.6435417    
## rcs(bmi)bmi''                                           0.1394 0.8891644    
## rcs(bmi)bmi'''                                         -0.4766 0.6336459    
## factor(rokare)1                                        63.0543 < 2.2e-16 ***
## factor(ami_history)1                                   23.0793 < 2.2e-16 ***
## factor(stroke_history)1                                37.7868 < 2.2e-16 ***
## factor(Civil)1                                        -17.5479 < 2.2e-16 ***
## factor(HushallsTyp_RTB)2                              -40.5020 < 2.2e-16 ***
## factor(HushallsTyp_RTB)3                               -6.1967 5.765e-10 ***
## factor(Sun2000niva_old)2                                3.2302 0.0012369 ** 
## factor(Sun2000niva_old)3                               -6.7643 1.340e-11 ***
## factor(Sun2000niva_old)4                               -8.7556 < 2.2e-16 ***
## factor(Sun2000niva_old)5                              -12.3042 < 2.2e-16 ***
## factor(Sun2000niva_old)6                              -16.6757 < 2.2e-16 ***
## factor(Sun2000niva_old)7                               -7.4632 8.442e-14 ***
## rcs(DispInkKEHB04)DispInkKEHB04                         0.5093 0.6105124    
## rcs(DispInkKEHB04)DispInkKEHB04'                       -3.3925 0.0006926 ***
## rcs(DispInkKEHB04)DispInkKEHB04''                      -0.3028 0.7620316    
## rcs(DispInkKEHB04)DispInkKEHB04'''                      4.1452 3.395e-05 ***
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk             5.2408 1.599e-07 ***
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk'           -0.1399 0.8887274    
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk''           0.0033 0.9974066    
## rcs(Raks_AndelArblosInk)Raks_AndelArblosInk'''          2.5236 0.0116155 *  
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk                 8.0624 7.482e-16 ***
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk'               -0.2092 0.8342954    
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk''               0.0258 0.9794224    
## rcs(Raks_AndelSjukInk)Raks_AndelSjukInk'''              1.4670 0.1423727    
## rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk               6.1973 5.743e-10 ***
## rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk'             -0.4884 0.6252830    
## rcs(Raks_AndelEkBisInk)Raks_AndelEkBisInk''             0.4190 0.6752011    
## factor(FodelseLand_EU27_2020)1                         13.8894 < 2.2e-16 ***
## factor(antithrombotic_agents)1                         40.8064 < 2.2e-16 ***
## factor(antihypertensive_comb)1                         -4.2767 1.897e-05 ***
## factor(lipid_modifying_agents)1                       -31.0160 < 2.2e-16 ***
## factor(hba1c_high)1                                    17.8313 < 2.2e-16 ***
## factor(GFR_high)1                                      -6.5774 4.786e-11 ***
## factor(kolesterol_high)1                               -7.6867 1.510e-14 ***
## factor(systoliskt_high)1                              -53.2273 < 2.2e-16 ***
## factor(cpap_treated)1:factor(antithrombotic_agents)1    0.8524 0.3939653    
## factor(cpap_treated)1:factor(antihypertensive_comb)1    0.3801 0.7038501    
## factor(cpap_treated)1:factor(lipid_modifying_agents)1   1.2519 0.2106174    
## rcs(alder)alder:rcs(bmi)bmi                             0.1338 0.8935754    
## rcs(alder)alder':rcs(bmi)bmi                            0.1920 0.8477293    
## rcs(alder)alder'':rcs(bmi)bmi                          -0.8295 0.4068372    
## rcs(alder)alder''':rcs(bmi)bmi                          1.8178 0.0690876 .  
## rcs(alder)alder:rcs(bmi)bmi'                           -0.3888 0.6973876    
## rcs(alder)alder':rcs(bmi)bmi'                           0.4019 0.6877352    
## rcs(alder)alder'':rcs(bmi)bmi'                          0.0390 0.9689088    
## rcs(alder)alder''':rcs(bmi)bmi'                        -0.3034 0.7616152    
## rcs(alder)alder:rcs(bmi)bmi''                           0.0115 0.9908011    
## rcs(alder)alder':rcs(bmi)bmi''                         -0.1160 0.9076809    
## rcs(alder)alder'':rcs(bmi)bmi''                        -0.0562 0.9551851    
## rcs(alder)alder''':rcs(bmi)bmi''                       -0.2755 0.7829592    
## rcs(alder)alder:rcs(bmi)bmi'''                          0.3268 0.7438456    
## rcs(alder)alder':rcs(bmi)bmi'''                        -0.1782 0.8585436    
## rcs(alder)alder'':rcs(bmi)bmi'''                        0.1187 0.9054991    
## rcs(alder)alder''':rcs(bmi)bmi'''                       0.6909 0.4896595    
## rcs(alder)alder:factor(GFR_high)1                       5.7658 8.126e-09 ***
## rcs(alder)alder':factor(GFR_high)1                     -0.1051 0.9163026    
## rcs(alder)alder'':factor(GFR_high)1                    -0.3415 0.7327490    
## rcs(alder)alder''':factor(GFR_high)1                   -0.0345 0.9725118    
## factor(cpap_treated)1:factor(GFR_high)1                -1.0389 0.2988498    
""",
]

print(format_table_dac(list_output))

# %%
