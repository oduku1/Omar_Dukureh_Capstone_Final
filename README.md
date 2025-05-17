# NBA Player Win Shares Machine Learning Project 

Omar Dukureh | Math 377 | Spring 2025

Table of Contents
- [PROJECT OVERVIEW](#project-overview)


# Project Overview
## Problem Area:
My area of interest is sports analytics, especially within the realm of basketball. One major challenge in this field is estimating how well a player will perform across an entire season. Traditional stats like points per game and field goal percentage are helpful but don't always give a full picture when projecting long-term outcomes. My project will aim to predict player win shares, which is a more comprehensive measure of a player's contribution to their team's success.

## Affected Stakeholders 
This project will benefit NBA teams, analysts, bettors, and users of fantasy basketball software. Teams can use these long-term projections to plan trades, rotations, and development strategies. Fantasy basketball players and sports bettors can gain an edge by anticipating breakout seasons or underperforming players. Analysts can use these models to explore broader trends and player trajectories.

## Proposed Data Science Approach:
Machine learning can significantly improve long-term performance prediction by analyzing historical data and identifying patterns that may not be obvious to human analysts. I plan to use regression models to forecast season stats such as points, assists, and rebounds based on prior season performances. I will try to complete this using XGBoost, a machine learning algorithm that deals with much more complex relationships. Unlike linear regression, its uses decision tree based learning and goes through every tree fixing the mistakes of the last. While similar models exist, many lack depth in statistical modeling and data handling and are not as fast as XGBoost.  

## The Impact:
The impact of this project will be significant. Teams can use these predictions to make informed decisions about player trades, rotations, and development strategies. Fantasy basketball players and sports bettors can gain an edge by anticipating breakout seasons or underperforming players. Analysts can use these models to explore broader trends and player trajectories.

## Dataset Description
 The dataset that I used for this is the NBA player Stats dataset from Basketball Reference, This includes data from the 2018-2019 NBA season to the 2024-2025 NBA season.
 The data includes player statistics such as points, rebounds, assists, and minutes played.
- Source : Basketball Reference
- Raw Data : /data/
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
  </tbody>
</table>


# Results
The project will generate a new dataframe with the following columns:
- Player: The name of the player.
- Asists: The number of assists made by the player.
- Steals: The number of steals made by the player.
- Blocks: The number of blocks made by the player.
- TRB: The total number of rebounds made by the player.
- TOV : The number of turnovers made by the player.
** These are seasson average stats for the player that the model is predicting ***

It will be a data frame that compares the predicted stats to the actual stats for the player.

## Limitations

- Limited time frames: The dataset only covers the 2020-2021 season to the 2023-2024 season.
- Small sample size: The dataset contains a limited number of players, which may limit the generalizability of the results.
- Non Specific Data: The dataset does not provide information about the specific team or coach that the player is on, which could be important for predicting player performance.
- Limited features: The dataset only includes minimal advanced statistics such as USG, TS%, etc, which may not be sufficient for accurate predictions for certain targets. 

## Future Work

- Expand dataset: Collect more data on player performance to improve the accuracy of the predictions.
- Incorporate additional features: Incorporate additional features such as team performance, coaching style, and player health to improve the accuracy of the predictions.
- Use more advanced machine learning techniques: Explore the use of more advanced machine learning techniques such as Deep learning to improve the accuracy of the predictions.
- Predict game-by-game performance: Predict player performance on a game-by-game basis to provide more detailed insights into player performance.

## References
SkyQuest. “Fantasy Sports Market Size & Share - Industry Growth: 2032.” Fantasy Sports Market Size & Share - Industry Growth | 2032, www.skyquestt.com/report/fantasy-sports-market#:~:text=Fantasy%20Sports%20Market%20size%20was,period%20(2025%2D2032). Accessed 30 Mar. 2025.

