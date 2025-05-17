# NBA Player Win Shares Machine Learning Project 

Omar Dukureh | Math 377 | Spring 2025

Table of Contents
- [PROJECT OVERVIEW](#project-overview)
- [DATASET DESCRIPTION](#dataset-description)
- [PROJECT STRUCTURE](#project-structure)


# Project Overview
## Problem Area:
My area of interest is sports analytics, especially within the realm of basketball. One major challenge in this field is estimating how well a player will perform across an entire season. Traditional stats like points per game and field goal percentage are helpful but don't always give a full picture when projecting long-term outcomes. My project will aim to predict player win shares, which is a more comprehensive measure of a player's contribution to their team's success.

## Affected Stakeholders 
This project will benefit NBA teams, analysts, bettors, and users of fantasy basketball software. Teams can use these long-term projections to plan trades, rotations, and development strategies. Fantasy basketball players and sports bettors can gain an edge by anticipating breakout seasons or underperforming players. Analysts can use these models to explore broader trends and player trajectories.

## Proposed Data Science Approach:
Machine learning can significantly improve long-term performance prediction by analyzing historical data and identifying patterns that may not be obvious to human analysts. I plan to use regression models to forecast season stats such as points, assists, and rebounds based on prior season performances. I will try to complete this using XGBoost, a machine learning algorithm that deals with much more complex relationships. Unlike linear regression, its uses decision tree based learning and goes through every tree fixing the mistakes of the last. While similar models exist, many lack depth in statistical modeling and data handling and are not as fast as XGBoost.  

## The Impact:
The impact of this project will be significant. Teams can use these predictions to make informed decisions about player trades, rotations, and development strategies. Fantasy basketball players and sports bettors can gain an edge by anticipating breakout seasons or underperforming players. Analysts can use these models to explore broader trends and player trajectories.

# Dataset Description
 The dataset that I used for this is the NBA player Stats dataset from Basketball Reference, This includes data from the 2018-2019 NBA season to the 2024-2025 NBA season.
 The data includes player statistics such as points, rebounds, assists, and minutes played.
- Source : Basketball Reference
- Raw Data : Located under data/raw/
- Processed Data : Located under data/processed/

### Data dictionary

<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Data Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
   <tr><td>Player</td><td>object</td><td>Name of the player</td></tr>
   <tr><td>Age</td><td>float64</td><td>Age of the player</td></tr>
   <tr><td>G</td><td>float64</td><td>Number of games played</td></tr>
    <tr><td>GS</td><td>float64</td><td>Games started</td></tr>
    <tr><td>MP</td><td>float64</td><td>Total minutes played</td></tr>
    <tr><td>FG</td><td>float64</td><td>Field goals made</td></tr>
    <tr><td>FGA</td><td>float64</td><td>Field goals attempted</td></tr>
    <tr><td>FG%</td><td>float64</td><td>Field goal percentage</td></tr>
    <tr><td>3P</td><td>float64</td><td>3-point field goals made</td></tr>
    <tr><td>3PA</td><td>float64</td><td>3-point field goals attempted</td></tr>
    <tr><td>3P%</td><td>float64</td><td>3-point field goal percentage</td></tr>
    <tr><td>2P</td><td>float64</td><td>2-point field goals made</td></tr>
    <tr><td>2PA</td><td>float64</td><td>2-point field goals attempted</td></tr>
    <tr><td>2P%</td><td>float64</td><td>2-point field goal percentage</td></tr>
    <tr><td>eFG%</td><td>float64</td><td>Effective field goal percentage</td></tr>
    <tr><td>FT</td><td>float64</td><td>Free throws made</td></tr>
    <tr><td>FTA</td><td>float64</td><td>Free throws attempted</td></tr>
    <tr><td>FT%</td><td>float64</td><td>Free throw percentage</td></tr>
    <tr><td>ORB</td><td>float64</td><td>Offensive rebounds</td></tr>
    <tr><td>DRB</td><td>float64</td><td>Defensive rebounds</td></tr>
    <tr><td>TRB</td><td>float64</td><td>Total rebounds</td></tr>
    <tr><td>AST</td><td>float64</td><td>Assists</td></tr>
    <tr><td>STL</td><td>float64</td><td>Steals</td></tr>
    <tr><td>BLK</td><td>float64</td><td>Blocks</td></tr>
    <tr><td>TOV</td><td>float64</td><td>Turnovers</td></tr>
    <tr><td>PF</td><td>float64</td><td>Personal fouls</td></tr>
    <tr><td>PTS</td><td>float64</td><td>Total points scored</td></tr>
    <tr><td>Year</td><td>int64</td><td>Season year</td></tr>
    <tr><td>WS</td><td>float64</td><td>Win shares</td></tr>
    <tr><td>VORP</td><td>float64</td><td>Value over replacement player</td></tr>
    <tr></td>DBPM<td>float64</td><td>Defensive box plus-minus</td></tr>
    <tr><td>PER</td><td>float64</td><td>Player efficiency rating</td></tr>
    <tr><td>BPM</td><td>float64</td><td>Box plus-minus</td></tr>
    <tr><td>USG%</td><td>float64</td><td>Usage Percentage</td></tr>
    <tr><td>STL%</td><td>float64</td><td>Steal Percentage</td></tr>
    <tr><td>TS%</td><td>float64</td><td>True Shooting Percentage</td></tr>
    <tr><td>PTS_per_TSA</td><td>float64</td><td>Points per True Shooting Attempt</td></tr>
    <tr><td>FT_rate</td><td>float64</td><td>Free Throw Rate</td></tr>
    <tr><td>3P_rate</td><td>float64</td><td>Three-Point Attempt Rate</td></tr>
    <tr><td>AST_TO</td><td>float64</td><td>Assist-to-Turnover Ratio</td></tr>
    <tr><td>TOV%</td><td>float64</td><td>Turnover Percentage</td></tr>
    <tr><td>TRB%_approx</td><td>float64</td><td>Approximate Total Rebound Percentage</td></tr>
    <tr><td>STL_per_min</td><td>float64</td><td>Steals per Minute</td></tr>
    <tr><td>BLK%</td><td>float64</td><td>Block Percentage</td></tr>
    <tr><td>Raw_EFF</td><td>float64</td><td>Raw Efficiency</td></tr>
    <tr><td>PF_per_min</td><td>float64</td><td>Personal Fouls per Minute</td></tr>
    <tr><td>Load</td><td>float64</td><td>Offensive Load</td></tr>
    <tr><td>Creation_proxy</td><td>float64</td><td>Shot Creation Proxy</td></tr>
    <tr><td>log_WS</td><td>float64</td><td>Logarithm of Win Shares</td></tr>
    <tr><td>OBPM</td><td>float64</td><td>Offensive Box Plus-Minus</td></tr>
    <tr><td>DBPM</td><td>float64</td><td>Defensive Box Plus-Minus</td></tr>
  </tbody>
</table>


# Project Structure
Capstone_377/
├── .venv/
├── data/
│ ├── raw/ # Immutable source data
│ └── processed/ # Cleaned, feature‑engineered data
│
├── notebooks/ # Jupyter notebooks
│ ├── 1_data_prep.ipynb
│ ├── 2_eda.ipynb
│ ├── 3_baseline_models.ipynb
│ └── 4_final_modeling.ipynb
│
├── outputs/ │ ├── models/
│ └── figures/
│ └── models/
├── slides/ # Slide deck PDF & source
│ ├── Capstone_Final_Presentation.pdf
│ └── Capstone_Final_Presentation.pptx


# Notebook & Modules 
| Feature       | Description                        | Status   | Notes                        |
|--------------|------------------------------------|----------|------------------------------|
| Data Cleaning| Remove nulls and outliers          | ✅ Done  | Used pandas                 |
| Model Train  | XGBoost Regression on 2024 stats   | ✅ Done  | 85% R² score                |
| Evaluation   | Test set RMSE                      | ⏳ In Progress | Need final test run     |
| Visualization| Matplotlib plots of predictions    | ❌ Not Started |                              |
| README       | Update with usage instructions     | ❌ Not Started | Add example usage            |


