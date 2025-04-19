# NBA Player Performance Prediction
## Problem Area:
My area of interest is sports analytics, especially within the realm of basketball. One major challenge in this field is estimating how well a player will perform across an entire season, considering factors like team trades, injuries, player development, and changing team dynamics. Traditional stats like points per game and field goal percentage are helpful but don't always give a full picture when projecting long-term outcomes. My project will aim to predict season-long stats—such as total points, rebounds, and assists—for NBA players in upcoming seasons.

## Affected Stakeholders 
This project will benefit NBA teams, analysts, bettors, and users of fantasy basketball software. Teams can use these long-term projections to plan trades, rotations, and development strategies. Fantasy basketball players and sports bettors can gain an edge by anticipating breakout seasons or underperforming players. Analysts can use these models to explore broader trends and player trajectories.

## Proposed Data Science Approach:
Machine learning can significantly improve long-term performance prediction by analyzing historical data and identifying patterns that may not be obvious to human analysts. I plan to use regression models to forecast season stats such as points, assists, and rebounds based on prior season performance, age, injury history, and team changes. I may also experiment with neural networks to capture more complex interactions, such as the impact of new teammates or changes in coaching strategies. While similar models exist, many lack depth in statistical modeling or fail to account for subtle but important factors like team chemistry or coaching philosophy.

## The Impact:
Accurate predictions of full-season player stats can have wide-reaching benefits. The fantasy sports industry alone is worth over $30.4 billion and is projected to reach $84.98 billion USD by 2032 (SkyQuest). Sportsbooks use predictive models to set odds, and more accurate forecasts can improve this process. NBA teams can use season-long predictions to make informed decisions around player contracts, rotations, and long-term strategy.

## Dataset Description
 The dataset that I used for this is the NBA Player Stats dataset from Basketball Reference, This includes data from the 2020-2021 season to the 2023-2024 season.
 The data includes player statistics such as points, rebounds, assists, and minutes played.

### Data dictionary
Column Name | Data Type | Description
Age | float64 | Age of the player
G | float64 | Number of games played
GS | float64 | Number of games started
MP | float64 | Minutes played
FG | float64 | Field goals made
FGA | float64 | Field goals attempted
FG% | float64 | Field goal percentage
3P | float64 | Three-point field goals made
3PA | float64 | Three-point field goals attempted
3P% | float64 | Three-point field goal percentage
2P | float64 | Two-point field goals made
2PA | float64 | Two-point field goals attempted
2P% | float64 | Two-point field goal percentage
eFG% | float64 | Effective field goal percentage
FT | float64 | Free throws made
FTA | float64 | Free throws attempted
FT% | float64 | Free throw percentage
ORB | float64 | Offensive rebounds
DRB | float64 | Defensive rebounds
TRB | float64 | Total rebounds
AST | float64 | Assists
STL | float64 | Steals
BLK | float64 | Blocks
TOV | float64 | Turnovers
PF | float64 | Personal fouls
PTS | float64 | Total points scored
Year | int64 | Season year
USG | float64 | Usage percentage — estimate of team plays used by the player while on floor
TS% | float64 | True shooting percentage
AST_TO | float64 | Assist-to-turnover ratio
REB_per_min | float64 | Rebounds per minute
PTS_per_FGA | float64 | Points scored per field goal attempt
FT_rate | float64 | Free throw rate (FTA per FGA)
3P_rate | float64 | Three-point attempt rate (3PA per FGA)
EFF | float64 | Player efficiency rating
TOV% | float64 | Turnover percentage (estimated percentage of possessions ending in a TO)
PF_per_min | float64 | Personal fouls per minute
STL_per_min | float64 | Steals per minute
BLK_per_min | float64 | Blocks per minute
TRB_rate | float64 | Rebound rate (estimated percentage of available rebounds grabbed)
PTS_log | float64 | Log-transformed points
AST_log | float64 | Log-transformed assists
TRB_log | float64 | Log-transformed total rebounds
STL_log | float64 | Log-transformed steals
BLK_log | float64 | Log-transformed blocks
TOV_log | float64 | Log-transformed turnovers

# Results
The project generates a new dataframe with the following columns:
- Player: The name of the player.
- Asists: The number of assists made by the player.
- Steals: The number of steals made by the player.
- Blocks: The number of blocks made by the player.
- TRB: The total number of rebounds made by the player.
- TOV : The number of turnovers made by the player.
** These are seasson average stats for the player that the model is predicting ***

## Limitations

- Limited time frames: The dataset only covers the 2020-2021 season to the 2023-2024 season.
- Small sample size: The dataset contains a limited number of players, which may limit the generalizability of the results.
- Non Specific Data: The dataset does not provide information about the specific team or coach that the player is on, which could be important for predicting player performance.


## Future Work

- Expand dataset: Collect more data on player performance to improve the accuracy of the predictions.
- Incorporate additional features: Incorporate additional features such as team performance, coaching style, and player health to improve the accuracy of the predictions.
- Use more advanced machine learning techniques: Explore the use of more advanced machine learning techniques such as deep learning to improve the accuracy of the predictions.
- Predict game-by-game performance: Predict player performance on a game-by-game basis to provide more detailed insights into player performance.

## References
SkyQuest. “Fantasy Sports Market Size & Share - Industry Growth: 2032.” Fantasy Sports Market Size & Share - Industry Growth | 2032, www.skyquestt.com/report/fantasy-sports-market#:~:text=Fantasy%20Sports%20Market%20size%20was,period%20(2025%2D2032). Accessed 30 Mar. 2025.

