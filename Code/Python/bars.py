#%%
import pandas as pd
import matplotlib.pyplot as plt


def plot_errors(df):
    # Calculate error bars
    errors = df["CPAP HR"] - df["95_low"], df["95_high"] - df["CPAP HR"]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(df.index, df["CPAP HR"], 
                yerr=errors, fmt='o', capsize=5, color='darkblue', ecolor='gray')

    plt.ylabel("Hazard Ratio (CPAP)")
    plt.xlabel("Variable Set to Baseline")
    plt.title("CPAP HR with 95% Confidence Intervals")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', color="lightgrey", alpha=0.75)
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    plt.show()


#%%
# Death group 2
df_death_g2 = pd.DataFrame([[5.776, 2.969, 11.244],
                            [6.001, 3.068, 11.74],
                            [3.978, 1.933, 8.187],
                            [4.000, 1.946, 8.224],
                            [4.104, 2.004, 8.404],
                            
                            [4.249, 2.078, 8.689],
                            ], 
                            index=["None", "+ HbA1c", "+ GFR", "+ Smoking", "+ SBP", "+ BMI"],
                            columns=["CPAP HR", "95_low", "95_high"]
                            )

plot_errors(df_death_g2)
# %%
# AMI group 1
df_ami_g1 = pd.DataFrame([[1.080, 0.456, 2.555],
                            [1.090, 0.458, 2.590],
                            [0.564, 0.213, 1.504],
                            [0.567, 0.213, 1.507],
                            [0.560, 0.211, 1.489],
                            [0.547, 0.206, 1.455],
                            [0.549, 0.206, 1.460],
                            ], 
                            index=["None", "+ HbA1c", "+ GFR", "+ Smoking", "+ Tot. cholesterol", "+ SBP", "+ BMI"],
                            columns=["CPAP HR", "95_low", "95_high"]
                            )

plot_errors(df_ami_g1)
# %%
# AMI group 2
df_ami_g2 = pd.DataFrame([[0.424, 0.259, 0.694],
                            [0.419, 0.256, 0.686],
                            [0.416, 0.254, 0.680],
                            [0.417, 0.255, 0.683],
                            [0.415, 0.254, 0.679],
                            [0.414, 0.253, 0.677],
                            ], 
                            index=["None", "+ HbA1c", "+ GFR", "+ Smoking", "+ Tot. cholesterol", "+ SBP"],
                            columns=["CPAP HR", "95_low", "95_high"]
                            )

plot_errors(df_ami_g2)
#%%
# Stroke group 1
df_stroke_g1 = pd.DataFrame([[1.070, 0.426, 2.694],
                            [1.069, 0.423, 2.701],
                            [1.383, 0.499, 3.828],
                            [1.379, 0.498, 3.817],
                            [1.372, 0.497, 3.790],
                            [1.343, 0.486, 3.713],
                            [1.345, 0.486, 3.720],
                            ], 
                            index=["None", "+ HbA1c", "+ GFR", "+ Smoking", "+ Tot. cholesterol", "+ SBP", "+ BMI"],
                            columns=["CPAP HR", "95_low", "95_high"]
                            )

plot_errors(df_stroke_g1)
#%%
# Stroke group 2
df_stroke_g2 = pd.DataFrame([[1.166, 0.874, 1.556],
                            [1.146, 0.859, 1.529],
                            [1.157, 0.867, 1.544],
                            [1.146, 0.859, 1.529],
                            [1.146, 0.859, 1.529],
                            ], 
                            index=["None", "+ HbA1c", "+ Smoking", "+ SBP", "+ BMI"],
                            columns=["CPAP HR", "95_low", "95_high"]
                            )

plot_errors(df_stroke_g2)