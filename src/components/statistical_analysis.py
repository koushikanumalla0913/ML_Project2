import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt
from src.exception import CustomException
from src.logger import logging
from scipy import stats

from src.utils import save_object


@dataclass
class StatisticalAnalysisConfig:
    StatisticalAnalysis_obj_file_path: str = os.path.join(
        "artifacts", "statisticalAnalysis.pkl"
    )


class StatisticalAnalysis:
    def __init__(self):
        self.config = StatisticalAnalysisConfig()

    def analyze_numeric_vs_target(self, file_path):
        try:
            # Read the data
            train_df = pd.read_csv(file_path)
            self.numeric_vars = [
                "age",
                "balance",
                "day",
                "duration",
                "campaign",
                "pdays",
                "previous",
            ]
            self.categorical_vars_nominal = [
                "job",
                "marital",
                "default",
                "housing",
                "loan",
                "contact",
                "month",
                "poutcome"
            ]
            self.categorical_vars_ordinal = ["education"]
            self.education_order = ["primary", "secondary", "tertiary"]
            train_df['education'] = pd.Categorical(train_df['education'], categories=self.education_order, ordered=True)
            self.target_col = "y"
            logging.info("StatisticalAnalysis initialized.")

            if self.target_col not in train_df.columns:
                logging.warning(
                    f"Target column {self.target_col} not found in dataframe."
                )

            for var in self.numeric_vars:
                if var not in train_df.columns:
                    logging.warning(f"Variable {var} not found in dataframe.")
                    continue

                # Plot boxplot
                logging.info(f"Plotting {var} vs {self.target_col}.")
                sns.boxplot(x=self.target_col, y=var, data=train_df)
                plt.title(f"{self.target_col} vs {var}")
                plt.xlabel(self.target_col)
                plt.ylabel(var)
                plt.tight_layout()
                plt.show()

                # T-test
                group1 = train_df[train_df[self.target_col] == "yes"][var]
                group2 = train_df[train_df[self.target_col] == "no"][var]
                t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy="omit")

                logging.info(
                    f"T-test for {var} vs {self.target_col}: t={t_stat:.4f}, p={p_val:.4f}"
                )

                if p_val <= 0.05:
                    print(f"{var}: Associated with {self.target_col}, p={p_val:.4f}")
                else:
                    print(
                        f"{var}: NOT associated with {self.target_col}, p={p_val:.4f}"
                    )

                # return file_path, self.config.StatisticalAnalysis_obj_file_path
            for var in self.categorical_vars_nominal:
                if var not in train_df.columns:
                    logging.warning(f"Variable {var} not found in dataframe.")
                    continue

                # Plot the mosaic plot (similar to R's mosaicplot)
                logging.info(f"Plotting mosaic plot for {var} vs {self.target_col}.")
                contingency_table = pd.crosstab(train_df[self.target_col], train_df[var])
                sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f"{var} vs {self.target_col}")
                plt.xlabel(var)
                plt.ylabel(self.target_col)
                plt.tight_layout()
                plt.show()

                # Perform Chi-Square Test
                chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

                logging.info(f"Chi-square test for {var} vs {self.target_col}: chi2={chi2_stat:.4f}, p={p_val:.4f}")

                if p_val <= 0.05:
                    print(f"{var}: Associated with {self.target_col}, p={p_val:.4f}")
                else:
                    print(f"{var}: NOT associated with {self.target_col}, p={p_val:.4f}")
            return file_path, self.config.StatisticalAnalysis_obj_file_path
        except Exception as e:
            logging.error(
                "Error occurred during numeric vs target analysis.", exc_info=True
            )
            raise CustomException(e, sys)
