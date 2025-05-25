#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

#%%
def format_thousands(x):
    return f"{round(x, -3) / 1000:.0f}k"


def latex_data_description(df_survival_times):
    # df_survival_times["age_at_ami"] = df_survival_times.alder + (df_survival_times.time_to_ami / 365.25)
    # df_survival_times["age_at_stroke"] = df_survival_times.alder + (df_survival_times.time_to_stroke / 365.25)
    # df_survival_times["age_at_death"] = df_survival_times.alder + (df_survival_times.time_to_death / 365.25)

    df_cpap_treated = df_survival_times[df_survival_times.cpap_treated == 1]
    df_cpap_untreated = df_survival_times[df_survival_times.cpap_treated == 0]

    df_description = pd.DataFrame(columns=["All", "CPAP treated", "CPAP untreated"])
    df_description.loc["Age, years (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.alder.mean():.1f} ({df_survival_times.alder.std():.1f})",
                                                                                        f"{df_cpap_treated.alder.mean():.1f} ({df_cpap_treated.alder.std():.1f})",
                                                                                        f"{df_cpap_untreated.alder.mean():.1f} ({df_cpap_untreated.alder.std():.1f})"
                                                                                        )
    df_description.loc["Women, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.sex.sum() / df_survival_times.sex.notnull().sum() * 100:.1f}",
                                                                                    f"{df_cpap_treated.sex.sum() / df_cpap_treated.sex.notnull().sum() * 100:.1f}",
                                                                                    f"{df_cpap_untreated.sex.sum() / df_cpap_untreated.sex.notnull().sum() * 100:.1f}"
                                                                                    )
    df_description.loc["Smoking, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.rokare.sum() / df_survival_times.rokare.notnull().sum() * 100:.1f}",
                                                                                        f"{df_cpap_treated.rokare.sum() / df_cpap_treated.rokare.notnull().sum() * 100:.1f}",
                                                                                        f"{df_cpap_untreated.rokare.sum() / df_cpap_untreated.rokare.notnull().sum() * 100:.1f}"
                                                                                        )
    df_description.loc[r"BMI, kg/m\textsuperscript{2} (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.bmi.mean():.1f} ({df_survival_times.bmi.std():.1f})",
                                                                                            f"{df_cpap_treated.bmi.mean():.1f} ({df_cpap_treated.bmi.std():.1f})",
                                                                                            f"{df_cpap_untreated.bmi.mean():.1f} ({df_cpap_untreated.bmi.std():.1f})"
                                                                                            )
    df_description.loc[r"GFR, mL/min/1.73m\textsuperscript{2} (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.GFR.mean():.1f} ({df_survival_times.GFR.std():.1f})",
                                                                                                f"{df_cpap_treated.GFR.mean():.1f} ({df_cpap_treated.GFR.std():.1f})",
                                                                                                f"{df_cpap_untreated.GFR.mean():.1f} ({df_cpap_untreated.GFR.std():.1f})"
                                                                                                )
    df_description.loc["HbA1c, mmol/mol (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.hba1c.mean():.1f} ({df_survival_times.hba1c.std():.1f})",
                                                                                                f"{df_cpap_treated.hba1c.mean():.1f} ({df_cpap_treated.hba1c.std():.1f})",
                                                                                                f"{df_cpap_untreated.hba1c.mean():.1f} ({df_cpap_untreated.hba1c.std():.1f})"
                                                                                                )
    df_description.loc["Cholesterol, mmol/L (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.kolesterol.mean():.1f} ({df_survival_times.kolesterol.std():.1f})",
                                                                                                f"{df_cpap_treated.kolesterol.mean():.1f} ({df_cpap_treated.kolesterol.std():.1f})",
                                                                                                f"{df_cpap_untreated.kolesterol.mean():.1f} ({df_cpap_untreated.kolesterol.std():.1f})"
                                                                                                )
    df_description.loc["SBP, mmHg (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.systoliskt.mean():.1f} ({df_survival_times.systoliskt.std():.1f})",
                                                                                                f"{df_cpap_treated.systoliskt.mean():.1f} ({df_cpap_treated.systoliskt.std():.1f})",
                                                                                                f"{df_cpap_untreated.systoliskt.mean():.1f} ({df_cpap_untreated.systoliskt.std():.1f})"
                                                                                                )
    df_description.loc["Antithrombotic agents, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.antithrombotic_agents.sum() / df_survival_times.antithrombotic_agents.notnull().sum() * 100:.1f}",
                                                                                                        f"{df_cpap_treated.antithrombotic_agents.sum() / df_cpap_treated.antithrombotic_agents.notnull().sum() * 100:.1f}",
                                                                                                        f"{df_cpap_untreated.antithrombotic_agents.sum() / df_cpap_untreated.antithrombotic_agents.notnull().sum() * 100:.1f}"
                                                                                                        )
    df_description.loc["Antihypertensive agents, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.antihypertensive_comb.sum() / df_survival_times.antihypertensive_comb.notnull().sum() * 100:.1f}",
                                                                                                        f"{df_cpap_treated.antihypertensive_comb.sum() / df_cpap_treated.antihypertensive_comb.notnull().sum() * 100:.1f}",
                                                                                                        f"{df_cpap_untreated.antihypertensive_comb.sum() / df_cpap_untreated.antihypertensive_comb.notnull().sum() * 100:.1f}"
                                                                                                        )
    df_description.loc["Lipid-modifying agents, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.lipid_modifying_agents.sum() / df_survival_times.lipid_modifying_agents.notnull().sum() * 100:.1f}",
                                                                                                        f"{df_cpap_treated.lipid_modifying_agents.sum() / df_cpap_treated.lipid_modifying_agents.notnull().sum() * 100:.1f}",
                                                                                                        f"{df_cpap_untreated.lipid_modifying_agents.sum() / df_cpap_untreated.lipid_modifying_agents.notnull().sum() * 100:.1f}"
                                                                                                        )
    df_description.loc["History of AMI, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.ami_history.sum() / df_survival_times.ami_history.notnull().sum() * 100:.1f}",
                                                                                                f"{df_cpap_treated.ami_history.sum() / df_cpap_treated.ami_history.notnull().sum() * 100:.1f}",
                                                                                                f"{df_cpap_untreated.ami_history.sum() / df_cpap_untreated.ami_history.notnull().sum() * 100:.1f}"
                                                                                                )
    df_description.loc["History of stroke, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.stroke_history.sum() / df_survival_times.stroke_history.notnull().sum() * 100:.1f}",
                                                                                                f"{df_cpap_treated.stroke_history.sum() / df_cpap_treated.stroke_history.notnull().sum() * 100:.1f}",
                                                                                                f"{df_cpap_untreated.stroke_history.sum() / df_cpap_untreated.stroke_history.notnull().sum() * 100:.1f}"
                                                                                                )
    df_description.loc["Widowed status, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.Civil.sum() / df_survival_times.Civil.notnull().sum() * 100:.1f}",
                                                                                          f"{df_cpap_treated.Civil.sum() / df_cpap_treated.Civil.notnull().sum() * 100:.1f}",
                                                                                          f"{df_cpap_untreated.Civil.sum() / df_cpap_untreated.Civil.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Living alone, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.HushallsTyp_RTB == 1].index) / df_survival_times.HushallsTyp_RTB.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.HushallsTyp_RTB == 1].index) / df_cpap_treated.HushallsTyp_RTB.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.HushallsTyp_RTB == 1].index) / df_cpap_untreated.HushallsTyp_RTB.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Living together, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.HushallsTyp_RTB == 2].index) / df_survival_times.HushallsTyp_RTB.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.HushallsTyp_RTB == 2].index) / df_cpap_treated.HushallsTyp_RTB.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.HushallsTyp_RTB == 2].index) / df_cpap_untreated.HushallsTyp_RTB.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Other living form, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.HushallsTyp_RTB == 3].index) / df_survival_times.HushallsTyp_RTB.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.HushallsTyp_RTB == 3].index) / df_cpap_treated.HushallsTyp_RTB.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.HushallsTyp_RTB == 3].index) / df_cpap_untreated.HushallsTyp_RTB.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Income prop. unemployment (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.Raks_AndelArblosInk.mean():.3f} ({df_survival_times.Raks_AndelArblosInk.std():.3f})",
                                                                                                f"{df_cpap_treated.Raks_AndelArblosInk.mean():.3f} ({df_cpap_treated.Raks_AndelArblosInk.std():.3f})",
                                                                                                f"{df_cpap_untreated.Raks_AndelArblosInk.mean():.3f} ({df_cpap_untreated.Raks_AndelArblosInk.std():.3f})"
                                                                                                )
    df_description.loc["Income prop. sickness (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.Raks_AndelSjukInk.mean():.3f} ({df_survival_times.Raks_AndelSjukInk.std():.3f})",
                                                                                                f"{df_cpap_treated.Raks_AndelSjukInk.mean():.3f} ({df_cpap_treated.Raks_AndelSjukInk.std():.3f})",
                                                                                                f"{df_cpap_untreated.Raks_AndelSjukInk.mean():.3f} ({df_cpap_untreated.Raks_AndelSjukInk.std():.3f})"
                                                                                                )
    df_description.loc["Income prop. financial aid (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.Raks_AndelEkBisInk.mean():.3f} ({df_survival_times.Raks_AndelEkBisInk.std():.3f})",
                                                                                                    f"{df_cpap_treated.Raks_AndelEkBisInk.mean():.3f} ({df_cpap_treated.Raks_AndelEkBisInk.std():.3f})",
                                                                                                    f"{df_cpap_untreated.Raks_AndelEkBisInk.mean():.3f} ({df_cpap_untreated.Raks_AndelEkBisInk.std():.3f})"
                                                                                                    )
    df_description.loc["Household disposable income, SEK (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.DispInkKEHB04.mean():.1f} ({df_survival_times.DispInkKEHB04.std():.1f})",
                                                                                                f"{df_cpap_treated.DispInkKEHB04.mean():.1f} ({df_cpap_treated.DispInkKEHB04.std():.1f})",
                                                                                                f"{df_cpap_untreated.DispInkKEHB04.mean():.1f} ({df_cpap_untreated.DispInkKEHB04.std():.1f})"
                                                                                                )
    df_description.loc["Preschool education, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.Sun2000niva_old == 1].index) / df_survival_times.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.Sun2000niva_old == 1].index) / df_cpap_treated.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.Sun2000niva_old == 1].index) / df_cpap_untreated.Sun2000niva_old.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Pre-secondary education, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.Sun2000niva_old.isin([2, 3])].index) / df_survival_times.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.Sun2000niva_old.isin([2, 3])].index) / df_cpap_treated.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.Sun2000niva_old.isin([2, 3])].index) / df_cpap_untreated.Sun2000niva_old.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Secondary education, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.Sun2000niva_old == 4].index) / df_survival_times.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.Sun2000niva_old == 4].index) / df_cpap_treated.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.Sun2000niva_old == 4].index) / df_cpap_untreated.Sun2000niva_old.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Post-secondary education, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.Sun2000niva_old.isin([5, 6])].index) / df_survival_times.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.Sun2000niva_old.isin([5, 6])].index) / df_cpap_treated.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.Sun2000niva_old.isin([5, 6])].index) / df_cpap_untreated.Sun2000niva_old.notnull().sum() * 100:.1f}"
                                                                                          )
    df_description.loc["Doctoral studies, \%", ["All", "CPAP treated", "CPAP untreated"]] = (f"{len(df_survival_times[df_survival_times.Sun2000niva_old == 7].index) / df_survival_times.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_treated[df_cpap_treated.Sun2000niva_old == 7].index) / df_cpap_treated.Sun2000niva_old.notnull().sum() * 100:.1f}",
                                                                                          f"{len(df_cpap_untreated[df_cpap_untreated.Sun2000niva_old == 7].index) / df_cpap_untreated.Sun2000niva_old.notnull().sum() * 100:.1f}"
                                                                                          )
    # df_description.loc["Age at AMI, years (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.age_at_ami.mean():.1f} ({df_survival_times.age_at_ami.std():.1f})",
    #                                                                                         f"{df_cpap_treated.age_at_ami.mean():.1f} ({df_cpap_treated.age_at_ami.std():.1f})",
    #                                                                                         f"{df_cpap_untreated.age_at_ami.mean():.1f} ({df_cpap_untreated.age_at_ami.std():.1f})"
    #                                                                                         )
    # df_description.loc["Age at stroke, years (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.age_at_stroke.mean():.1f} ({df_survival_times.age_at_stroke.std():.1f})",
    #                                                                                         f"{df_cpap_treated.age_at_stroke.mean():.1f} ({df_cpap_treated.age_at_stroke.std():.1f})",
    #                                                                                         f"{df_cpap_untreated.age_at_stroke.mean():.1f} ({df_cpap_untreated.age_at_stroke.std():.1f})"
    #                                                                                         )
    # df_description.loc["Age at death, years (SD)", ["All", "CPAP treated", "CPAP untreated"]] = (f"{df_survival_times.age_at_death.mean():.1f} ({df_survival_times.age_at_death.std():.1f})",
    #                                                                                         f"{df_cpap_treated.age_at_death.mean():.1f} ({df_cpap_treated.age_at_death.std():.1f})",
    #                                                                                         f"{df_cpap_untreated.age_at_death.mean():.1f} ({df_cpap_untreated.age_at_death.std():.1f})"
    #                                                                                         )

    df_description = df_description.rename(columns={"All": f"\\bfseries{{\makecell{{All\\\\n={len(df_survival_times.index)}}}}}",
                                                    "CPAP treated": f"\\bfseries{{\makecell{{CPAP treated\\\\n={len(df_cpap_treated.index)}}}}}",
                                                    "CPAP untreated": f"\\bfseries{{\makecell{{CPAP untreated\\\\n={len(df_cpap_untreated.index)}}}}}"})

    print(df_description)
    print(df_description.to_latex())


def fit_kaplan_meier_curves(df_survival_times):
    kmf = KaplanMeierFitter()
    df_untreated = df_survival_times[df_survival_times.cpap_treated == 0]
    df_treated = df_survival_times[df_survival_times.cpap_treated == 1]

    # Run KM for each type of event
    for event_type in ["ami", "stroke", "cv_death", "death"]:
        if event_type == "ami":
            title = event_type.upper()
        elif event_type in ["stroke", "death"]:
            title = event_type.capitalize()
        else:
            title = event_type

        # Stratified by CPAP treatment
        # Fit and plot KM for untreated group
        plt.figure(figsize=(8, 6))
        kmf.fit(df_untreated[f"time_to_{event_type}"] / 365.25, df_untreated[event_type], label="untreated")
        kmf.plot()

        # Fit and plot KM for treated group
        kmf.fit(df_treated[f"time_to_{event_type}"] / 365.25, df_treated[event_type], label="treated")
        kmf.plot()
        plt.title(title)
        plt.ylabel("Probability of survival")
        plt.xlabel("Time (years)")
        plt.legend()
        plt.show()

        # Perform logrank test
        res = logrank_test(df_untreated[f"time_to_{event_type}"], df_treated[f"time_to_{event_type}"], df_untreated[event_type], df_treated[event_type])
        print(f"Log-rank test results for {event_type}:")
        res.print_summary()

        # Print group sizes
        print("No. people in untreated group with event: ", df_survival_times.loc[df_survival_times.cpap_treated == 0, event_type].sum())
        print("No. people in treated group with event: ", df_survival_times.loc[df_survival_times.cpap_treated == 1, event_type].sum())
        print("Total no. people in untreated group: ", df_survival_times.loc[df_survival_times.cpap_treated == 0, "LopNr"].nunique())
        print("Total no. people in treated group: ", df_survival_times.loc[df_survival_times.cpap_treated == 1, "LopNr"].nunique())

        # Stratified by gender
        plt.figure(figsize=(8, 6))
        kmf.fit(df_survival_times.loc[df_survival_times.sex == 1, f"time_to_{event_type}"] / 365.25, df_survival_times.loc[df_survival_times.sex == 1, event_type], label="female")
        kmf.plot()

        kmf.fit(df_survival_times.loc[df_survival_times.sex == 0, f"time_to_{event_type}"] / 365.25, df_survival_times.loc[df_survival_times.sex == 0, event_type], label="male")
        kmf.plot()
        plt.title(title)
        plt.ylabel("Probability of survival")
        plt.xlabel("Time (years)")
        plt.legend()
        plt.show()

        # Stratified by gender and CPAP treatment
        plt.figure(figsize=(8, 6))
        # Untreated
        kmf.fit(df_untreated.loc[df_untreated.sex == 1, f"time_to_{event_type}"] / 365.25, df_untreated.loc[df_untreated.sex == 1, event_type], label="female, untreated")
        kmf.plot()

        kmf.fit(df_untreated.loc[df_untreated.sex == 0, f"time_to_{event_type}"] / 365.25, df_untreated.loc[df_untreated.sex == 0, event_type], label="male, untreated")
        kmf.plot()

        # Treated
        kmf.fit(df_treated.loc[df_treated.sex == 1, f"time_to_{event_type}"] / 365.25, df_treated.loc[df_treated.sex == 1, event_type], label="female, treated")
        kmf.plot()

        kmf.fit(df_treated.loc[df_treated.sex == 0, f"time_to_{event_type}"] / 365.25, df_treated.loc[df_treated.sex == 0, event_type], label="male, treated")
        kmf.plot()
        
        plt.title(title)
        plt.ylabel("Probability of survival")
        plt.xlabel("Time (years)")
        plt.legend()
        plt.show()


def plot_treatment_counts_over_time(df):
    # Ensure times are sorted
    df_sorted = df.sort_values(by=['start', 'stop'])

    # Create a list of all time change events
    events = []

    for _, row in df_sorted.iterrows():
        if row['cpap_treated'] == 1:
            events.append((row['start'], +1))  # treatment starts
            events.append((row['stop'], -1))   # treatment ends

    # Create DataFrame and compute cumulative sum
    events_df = pd.DataFrame(events, columns=['time', 'change'])
    events_df = events_df.sort_values('time')

    events_df['n_treated'] = events_df['change'].cumsum()

    # Drop duplicates if multiple changes happen at the same time
    events_df = events_df.drop(columns='change')

    # Plot
    plt.figure(figsize=(10, 5))
    plt.step(events_df['time'], events_df['n_treated'], where='post')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of People on Treatment')
    plt.title('Number of Treated Individuals Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#%%
df_survival_times = pd.read_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_survival_times.csv")
df_survival_times_sub = pd.read_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times.csv")

latex_data_description(df_survival_times)
latex_data_description(df_survival_times_sub)
#%%

print("--- ALL OF NDR ---")
fit_kaplan_meier_curves(df_survival_times)

print("--- NDR + SESAR INTERSECTION ---")
fit_kaplan_meier_curves(df_survival_times_sub)

#%%

fig, ax = plt.subplots()
sns.boxplot(data=df_survival_times_sub, x="cpap_treated", y="IV_AHI")
y_min, y_max = ax.get_ylim()
ax.axhspan(y_min, 5, color="green", alpha=0.6)
ax.axhspan(5, 15, color="yellow", alpha=0.6)
ax.axhspan(15, 30, color="orange", alpha=0.6)
ax.axhspan(30, y_max, color="red", alpha=0.6)
ax.set_ylim(y_min, y_max)

fig, ax = plt.subplots()
sns.boxplot(data=df_survival_times_sub, x="cpap_treated", y="IV_ODI")
y_min, y_max = ax.get_ylim()
ax.axhspan(y_min, 5, color="green", alpha=0.6)
ax.axhspan(5, 15, color="yellow", alpha=0.6)
ax.axhspan(15, 30, color="orange", alpha=0.6)
ax.axhspan(30, y_max, color="red", alpha=0.6)
ax.set_ylim(y_min, y_max)

fig, ax = plt.subplots()
sns.boxplot(data=df_survival_times_sub, x="cpap_treated", y="IV_AverageSaturation")
y_min, y_max = ax.get_ylim()
ax.axhspan(y_min, 5, color="green", alpha=0.6)
ax.axhspan(5, 15, color="yellow", alpha=0.6)
ax.axhspan(15, 30, color="orange", alpha=0.6)
ax.axhspan(30, y_max, color="red", alpha=0.6)
ax.set_ylim(y_min, y_max)

# %%

df_death_tv = pd.read_csv(f"/vault/marja987_amed/subsets_thesis_marijn/Data/death_tv.csv")
plot_treatment_counts_over_time(df_death_tv)
# %%

def compute_95_ci_hr(coef, se):
    coef = float(coef)
    se = float(se)
    lower = np.exp(coef - 1.96 * se)
    upper = np.exp(coef + 1.96 * se)

    print(f"{np.exp(coef):.3f} ({lower:.3f}, {upper:.3f})")


compute_95_ci_hr("4.979e-01", "1.034e-01")

#%%

df_survival_times_sub = pd.read_csv("/vault/marja987_amed/subsets_thesis_marijn/Data/ndr_sesar_survival_times.csv")

df_survival_times_sub = df_survival_times_sub.sort_values("time_to_death")
df_treated = df_survival_times_sub[df_survival_times_sub.cpap_treated == 1]
df_untreated = df_survival_times_sub[df_survival_times_sub.cpap_treated == 0]

print(len(df_treated.index))
print(df_treated[["LopNr", "time_to_death"]][:15])

print(len(df_untreated.index))
print(df_untreated[["LopNr", "time_to_death"]][:15])

df_km_ex = pd.DataFrame(columns=["A", "B", "C", "D"])
n_risk_set_treated = 130
n_risk_set_untreated = 100
for index, row in df_treated[:15].reset_index(drop=True).iterrows():
    df_km_ex.loc[index, "A"] = row["time_to_death"]
    n_failed = len(df_treated[df_treated.time_to_death == row["time_to_death"]].index)

    if index == 0:
        df_km_ex.loc[index, "B"] = f"$1\\times(1-{n_failed}/{n_risk_set_treated})={1 * (1 - n_failed / n_risk_set_treated):.3f}$"
        n_risk_set_treated -= n_failed
    else:
        prev_s_t = float(df_km_ex.loc[index - 1, "B"][-6:-1])
        df_km_ex.loc[index, "B"] = f"$S({df_km_ex.iloc[index - 1, 0]})\\times(1-{n_failed}/{n_risk_set_treated})={prev_s_t * (1 - n_failed / n_risk_set_treated):.3f}$"
        
        if df_km_ex.loc[index, "A"] != df_km_ex.loc[index - 1, "A"]:
            n_risk_set_treated -= n_failed


cols = pd.MultiIndex.from_tuples([
    (r"\bfseries{\gls{cpap} treated}", r"\bfseries{$t_j$ (days)}"),
    (r"\bfseries{\gls{cpap} treated}", r"\bfseries{\gls{km} survival function $S(t_j)$}"),
    (r"\bfseries{Untreated}", r"\bfseries{$t_j$ (days)}"),
    (r"\bfseries{Untreated}", r"\bfseries{\gls{km} survival function $S(t_j)$}"),
])

# %%
