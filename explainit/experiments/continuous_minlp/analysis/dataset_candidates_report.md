# Candidate datasets for the continuous-target MINLP experiment

## Goal

This report focuses on datasets that are good not only for regression, but also
for **counterfactual and priority-based reasoning**. For each dataset, I want to
know:

- what the continuous target means,
- what features are used to predict it,
- why someone would care about predicting it,
- what human decision scenario could motivate priorities,
- whether the dataset is easy to download and easy to integrate.

The target criteria remain:

- continuous target,
- mix of numerical and categorical features,
- preferably no more than about 20 usable predictors,
- preferably many rows,
- limited preprocessing,
- easy download from a stable public source.

## Recommended first 10

These are still the best first-pass choices for your experiment:

1. `diamonds`
2. `bike_sharing_hourly`
3. `abalone`
4. `insurance_charges`
5. `wage`
6. `credit`
7. `college`
8. `carseats`
9. `cps85`
10. `auto_mpg`

## Compact comparison table

| Dataset | Rows | Target | What is being predicted | Why predict it | Priority-scenario quality | Download ease |
| --- | ---: | --- | --- | --- | --- | --- |
| Diamonds | 53,940 | `price` | diamond market value | pricing and appraisal | Strong | Excellent |
| Bike Sharing (hourly) | 17,379 | `cnt` | rented bikes per hour | staffing and supply planning | Strong | Excellent |
| Abalone | 4,177 | `Rings` | age proxy | harvest and biology decisions | Medium | Excellent |
| Insurance Charges | 1,338 | `charges` | medical cost | pricing and risk estimation | Strong | Good |
| Wage | 3,000 | `wage` | hourly wage | labor economics and compensation | Strong | Excellent |
| Credit | 400 | `Balance` | credit card balance | banking risk and customer finance | Strong | Excellent |
| College | 777 | `Outstate` or `Grad.Rate` | tuition or graduation outcome | planning and benchmarking | Medium | Excellent |
| Carseats | 400 | `Sales` | store sales | retail planning | Strong | Excellent |
| CPS85 | 534 | `wage` | wage | labor market analysis | Strong | Excellent |
| Auto MPG | 398 | `mpg` | fuel efficiency | vehicle design and consumer choice | Strong | Excellent |
| Forest Fires | 517 | `area` | burned area | wildfire risk response | Strong | Excellent |
| Servo | 167 | `class` | servo response time | control-system tuning | Strong | Excellent |
| Palmer Penguins | 344 | `body_mass_g` | penguin body mass | ecological analysis | Medium | Excellent |
| Hitters | 322 | `Salary` | player salary | team budgeting | Strong | Excellent |
| Tips | 244 | `tip` | gratuity amount | restaurant behavior analysis | Medium | Excellent |
| Salaries | 397 | `salary` | professor salary | compensation analysis | Strong | Excellent |
| Brazilian Houses to Rent | 10,692 | rent amount | monthly rent | housing price estimation | Strong | Good |
| House Sales in King County | 21,613 | `price` | sale price | valuation and pricing | Strong | Good |

## Detailed dataset notes

## 1. Diamonds

- **Source:** `seaborn` or OpenML
- **Rows / predictors:** 53,940 rows, 9 usable predictors
- **Target:** `price`
- **What the target means:** sale price of a diamond
- **Features:**
  - **Categorical:** `cut`, `color`, `clarity`
  - **Numerical:** `carat`, `depth`, `table`, `x`, `y`, `z`
- **Why someone would want to predict it:** jewelers, insurers, and online marketplaces want a quick estimate of market value from measurable stone properties.
- **Natural scenario for your work:** a jeweler wants to understand what changes in a diamond profile would move it toward a lower or higher price segment while preserving realism.
- **Priority ideas:** prefer changing quality-related categories less than physical size, or the reverse. You could model scenarios like "keep carat nearly fixed but prefer better cut", or "achieve lower price mostly by reducing dimensions, not clarity".
- **Why good for MINLP priorities:** clear semantics, mixed discrete and continuous attributes, and many plausible preference structures.
- **Download:** `seaborn.load_dataset("diamonds")`

## 2. Bike Sharing (hourly)

- **Source:** UCI
- **Rows / predictors:** 17,379 rows, about 12 usable predictors after dropping leakage columns
- **Target:** `cnt`
- **What the target means:** total number of rented bikes in a given hour
- **Features:**
  - **Categorical / discrete:** `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`
  - **Numerical:** `temp`, `atemp`, `hum`, `windspeed`
- **Why someone would want to predict it:** operators need demand forecasts for bike allocation, staffing, maintenance, and rebalancing.
- **Natural scenario for your work:** a city mobility planner wants to understand what conditions are associated with lower demand and what interventions or contexts correspond to a target demand level.
- **Priority ideas:** prefer changes in weather-sensitive or calendar features depending on the story. For example, "reach lower demand mostly through bad weather and off-peak hours rather than holidays", or "find a nearby higher-demand hour with similar season and weather".
- **Important note:** drop `instant`, `dteday`, `casual`, and `registered`. `casual` and `registered` leak the target because they sum to `cnt`.
- **Why good for MINLP priorities:** very interpretable, strong operational story, large sample size.
- **Download:** UCI dataset `275`, easy with `ucimlrepo`

## 3. Abalone

- **Source:** UCI
- **Rows / predictors:** 4,177 rows, 8 predictors
- **Target:** `Rings`
- **What the target means:** number of shell rings, commonly used as a proxy for age
- **Features:**
  - **Categorical:** `Sex`
  - **Numerical:** length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
- **Why someone would want to predict it:** marine biologists and fisheries can use it to estimate age without destructive measurement.
- **Natural scenario for your work:** a fisheries analyst wants to understand what physical traits correspond to younger or older abalones.
- **Priority ideas:** prefer keeping sex fixed, allow moderate changes in mass-related attributes, or prefer changes that mostly affect shell dimensions instead of organ-weight variables.
- **Why good for MINLP priorities:** straightforward physical variables and an easy story, though only one categorical feature.
- **Download:** UCI dataset `1`

## 4. Insurance Charges

- **Source:** public CSV mirrors
- **Rows / predictors:** 1,338 rows, 6 predictors
- **Target:** `charges`
- **What the target means:** individual medical insurance cost
- **Features:**
  - **Categorical:** `sex`, `smoker`, `region`
  - **Numerical:** `age`, `bmi`, `children`
- **Why someone would want to predict it:** insurers want cost estimates, and policy analysts want to understand what drives expenses.
- **Natural scenario for your work:** an insurance advisor wants to show a client what profile changes are associated with reduced expected medical costs.
- **Priority ideas:** strongly prefer non-actionable `sex` and `region`, permit only decreasing `bmi`, and maybe keep `children` fixed. This gives a very natural actionable counterfactual story.
- **Why good for MINLP priorities:** small, clean, and one of the clearest actionability-driven datasets.
- **Download:** simple CSV mirror, for example StatLect

## 5. Wage

- **Source:** `Rdatasets`, `ISLR::Wage`
- **Rows / predictors:** 3,000 rows, about 9 usable predictors
- **Target:** `wage`
- **What the target means:** worker hourly wage
- **Features:**
  - **Categorical:** `maritl`, `race`, `education`, `region`, `jobclass`, `health`, `health_ins`
  - **Numerical:** `year`, `age`
- **Why someone would want to predict it:** labor economists, policy analysts, or career advisors may want to explain wage differences.
- **Natural scenario for your work:** a worker or policy analyst wants to understand what profile factors are associated with moving to a higher wage bracket.
- **Priority ideas:** immutable demographic traits can be pinned, while education or job class can be prioritized differently depending on the story. A counterfactual could ask for a higher wage while preferring education changes over health-status changes.
- **Important note:** drop `logwage` if the target is `wage`, because it is derived from the target.
- **Why good for MINLP priorities:** very rich categorical structure and easy human interpretation.
- **Download:** direct CSV from `Rdatasets`

## 6. Credit

- **Source:** `Rdatasets`, `ISLR::Credit`
- **Rows / predictors:** 400 rows, 10 predictors
- **Target:** `Balance`
- **What the target means:** average outstanding credit card balance
- **Features:**
  - **Categorical:** `Gender`, `Student`, `Married`, `Ethnicity`
  - **Numerical:** `Income`, `Limit`, `Rating`, `Cards`, `Age`, `Education`
- **Why someone would want to predict it:** banks want to understand consumer debt exposure and customer segments.
- **Natural scenario for your work:** a financial advisor wants to find the profile changes associated with lower expected revolving balance.
- **Priority ideas:** keep demographic traits fixed, prefer changing credit limit or rating less than income, or vice versa depending on whether the viewpoint is bank-centric or customer-centric.
- **Why good for MINLP priorities:** extremely clean and naturally supports actionable versus non-actionable features.
- **Download:** direct CSV from `Rdatasets`

## 7. College

- **Source:** `Rdatasets`, `ISLR::College`
- **Rows / predictors:** 777 rows, 17 usable predictors
- **Target options:** `Outstate` or `Grad.Rate`
- **What the target means:**
  - `Outstate`: out-of-state tuition,
  - `Grad.Rate`: graduation rate.
- **Features:**
  - **Categorical:** `Private`
  - **Numerical:** enrollment, application, faculty, expenditure, alumni, and student-body statistics
- **Why someone would want to predict it:** education researchers or administrators may want to understand how institutional characteristics relate to pricing or outcomes.
- **Natural scenario for your work:** a university planner wants to understand which institutional profile is associated with a higher graduation rate or lower tuition.
- **Priority ideas:** keep `Private` fixed, prefer changes in spending or faculty ratio rather than admissions volume, or emphasize student-support variables.
- **Why good for MINLP priorities:** the scenario is realistic, though only one categorical feature.
- **Download:** direct CSV from `Rdatasets`

## 8. Carseats

- **Source:** `Rdatasets`, `ISLR::Carseats`
- **Rows / predictors:** 400 rows, 10 predictors
- **Target:** `Sales`
- **What the target means:** unit sales of child car seats at a store
- **Features:**
  - **Categorical:** `ShelveLoc`, `Urban`, `US`
  - **Numerical:** `CompPrice`, `Income`, `Advertising`, `Population`, `Price`, `Age`, `Education`
- **Why someone would want to predict it:** retailers want to understand what store and market conditions drive sales.
- **Natural scenario for your work:** a regional manager wants to know what store profile changes are associated with higher sales.
- **Priority ideas:** prefer changing price and advertising before changing location-related categories, or insist that `Urban` and `US` stay fixed and let only pricing and local-market attributes move.
- **Why good for MINLP priorities:** very clear business story and several natural actionability choices.
- **Download:** direct CSV from `Rdatasets`

## 9. CPS85

- **Source:** `Rdatasets` or OpenML
- **Rows / predictors:** 534 rows, about 10 predictors
- **Target:** `wage`
- **What the target means:** hourly wage from labor survey data
- **Features:**
  - **Categorical:** `sex`, `sector`, `union`, `married`, `race`, `south`
  - **Numerical:** `education`, `experience`, `age`
- **Why someone would want to predict it:** economists and policy analysts study drivers of pay disparities.
- **Natural scenario for your work:** a labor-policy analyst wants to understand which worker profiles correspond to higher wages while respecting immutable traits.
- **Priority ideas:** lock demographics, allow education to increase only, prefer sector or union changes depending on the explanatory story.
- **Why good for MINLP priorities:** very natural notion of immutable and directional features.
- **Download:** direct CSV from `Rdatasets`

## 10. Auto MPG

- **Source:** UCI
- **Rows / predictors:** 398 rows, around 7 core predictors
- **Target:** `mpg`
- **What the target means:** miles per gallon, a direct measure of fuel efficiency
- **Features:**
  - **Categorical / discrete:** `origin`, optionally `cylinders`
  - **Numerical:** `displacement`, `horsepower`, `weight`, `acceleration`, `model year`
- **Why someone would want to predict it:** manufacturers and buyers care about vehicle efficiency.
- **Natural scenario for your work:** an engineer wants to know what design changes would correspond to improved fuel efficiency.
- **Priority ideas:** prefer reducing weight over changing origin, or prefer smaller horsepower reductions than large displacement changes.
- **Caveat:** some versions contain missing horsepower values, and `car name` is usually dropped.
- **Why good for MINLP priorities:** one of the clearest engineering-style counterfactual datasets.
- **Download:** UCI dataset `9`

## 11. Forest Fires

- **Source:** UCI
- **Rows / predictors:** 517 rows, 12 predictors
- **Target:** `area`
- **What the target means:** burned forest area
- **Features:**
  - **Categorical:** `month`, `day`
  - **Numerical:** `X`, `Y`, `FFMC`, `DMC`, `DC`, `ISI`, `temp`, `RH`, `wind`, `rain`
- **Why someone would want to predict it:** fire managers want to estimate fire severity from weather and seasonal conditions.
- **Natural scenario for your work:** an emergency planner wants to understand what condition profile corresponds to a smaller fire.
- **Priority ideas:** prefer keeping location fixed, allow weather variables to change, or prefer seasonal shifts over extreme weather changes.
- **Caveat:** target is highly skewed, so modeling may be harder than the schema suggests.
- **Download:** UCI dataset `162`

## 12. Servo

- **Source:** UCI
- **Rows / predictors:** 167 rows, 4 predictors
- **Target:** `class`
- **What the target means:** rise time or response of a servo mechanism
- **Features:**
  - **Categorical:** `motor`, `screw`
  - **Numerical:** `pgain`, `vgain`
- **Why someone would want to predict it:** engineers want to tune a control system for desired behavior.
- **Natural scenario for your work:** a control engineer wants to reduce response time while preferring gain adjustments over hardware substitutions.
- **Priority ideas:** hardware type can be expensive to change, so you can strongly prefer tuning `pgain` and `vgain` before switching `motor` or `screw`.
- **Why good for MINLP priorities:** tiny but almost ideal for priority design experiments.
- **Download:** UCI dataset `87`

## 13. Palmer Penguins

- **Source:** `palmerpenguins` or `Rdatasets`
- **Rows / predictors:** 344 rows, about 7 predictors
- **Target:** `body_mass_g`
- **What the target means:** penguin body mass in grams
- **Features:**
  - **Categorical:** `species`, `island`, `sex`
  - **Numerical:** bill length, bill depth, flipper length, `year`
- **Why someone would want to predict it:** ecologists study how species and morphology relate to body mass.
- **Natural scenario for your work:** a biologist wants to understand how morphological measurements explain heavier or lighter penguins.
- **Priority ideas:** keep species fixed, prefer modest changes in bill and flipper measurements, treat island as contextual and non-actionable.
- **Why good for MINLP priorities:** scientifically interpretable, though not as naturally actionable as business or medical datasets.
- **Download:** package dataset or direct CSV

## 14. Hitters

- **Source:** `Rdatasets`, `ISLR::Hitters`
- **Rows / predictors:** 322 rows, about 19 predictors
- **Target:** `Salary`
- **What the target means:** baseball player salary
- **Features:**
  - **Categorical:** `League`, `Division`, `NewLeague`
  - **Numerical:** current-season and career batting statistics
- **Why someone would want to predict it:** teams and analysts may study what performance profiles command higher salaries.
- **Natural scenario for your work:** a team analyst wants to see what performance trajectory corresponds to a target salary.
- **Priority ideas:** prefer improving current-season stats over changing league context, or prioritize career consistency over single-year spikes.
- **Caveat:** missing salary rows need filtering.
- **Download:** direct CSV from `Rdatasets`

## 15. Tips

- **Source:** `seaborn`
- **Rows / predictors:** 244 rows, 6 predictors
- **Target:** `tip`
- **What the target means:** tip amount left by a restaurant customer
- **Features:**
  - **Categorical:** `sex`, `smoker`, `day`, `time`
  - **Numerical:** `total_bill`, `size`
- **Why someone would want to predict it:** behavioral or service researchers may study gratuity patterns.
- **Natural scenario for your work:** a restaurant manager wants to understand which table contexts are associated with higher tips.
- **Priority ideas:** keep day/time fixed, prefer bill-size changes over demographic changes, or focus on party size.
- **Why good for MINLP priorities:** easy to understand, but small and less substantive than other options.
- **Download:** `seaborn.load_dataset("tips")`

## 16. Salaries

- **Source:** `Rdatasets`, `carData::Salaries`
- **Rows / predictors:** 397 rows, 5 predictors
- **Target:** `salary`
- **What the target means:** professor annual salary
- **Features:**
  - **Categorical:** `rank`, `discipline`, `sex`
  - **Numerical:** `yrs.since.phd`, `yrs.service`
- **Why someone would want to predict it:** universities analyze pay equity and career progression.
- **Natural scenario for your work:** an academic HR analyst wants to understand what profile differences correspond to higher salary.
- **Priority ideas:** `sex` is non-actionable, rank changes can be expensive, years-since-PhD can only increase, and years of service can be directional.
- **Why good for MINLP priorities:** very clean, realistic, and easy to narrate.
- **Download:** direct CSV from `Rdatasets`

## 17. Brazilian Houses to Rent

- **Source:** OpenML or public mirrors
- **Rows / predictors:** 10,692 rows, about 12 predictors
- **Target:** rent amount, preferably base rent rather than total cost
- **What the target means:** monthly rental price of a property
- **Features:**
  - **Categorical:** `city`, `animal`, `furniture`, maybe `floor` if kept categorical
  - **Numerical:** `area`, `rooms`, `bathroom`, `parking spaces`, condo fee, fire insurance, tax-like columns
- **Why someone would want to predict it:** renters, landlords, and platforms want to estimate fair rental value.
- **Natural scenario for your work:** a renter wants to know what property characteristics correspond to a target rent while preserving preferred city or furnishing status.
- **Priority ideas:** keep city fixed, prefer reducing area or optional amenities before changing room count, or keep pet-allowance fixed.
- **Caveat:** choose predictors carefully so the target is not too directly reconstructed from fee columns.
- **Download:** OpenML or public CSV mirror

## 18. House Sales in King County

- **Source:** OpenML or public mirrors
- **Rows / predictors:** 21,613 rows, about 18-19 usable predictors
- **Target:** `price`
- **What the target means:** house sale price
- **Features:**
  - **Categorical / discrete:** `waterfront`, `view`, `condition`, `grade`, `zipcode`
  - **Numerical:** bedrooms, bathrooms, floors, sqft features, latitude, longitude, year built, year renovated
- **Why someone would want to predict it:** buyers, sellers, and automated valuation systems need price estimates.
- **Natural scenario for your work:** a buyer wants to understand which house characteristics are associated with a lower target price while keeping location nearly fixed.
- **Priority ideas:** strongly prefer keeping `zipcode` fixed, allow moderate changes in size and quality variables, and maybe treat waterfront as almost immutable.
- **Caveat:** drop identifiers and parse or remove date.
- **Why good for MINLP priorities:** excellent real-estate counterfactual story and large sample size.
- **Download:** OpenML or public mirror

## Best datasets for scenario-driven priorities

If your paper needs strong human stories for both prediction and preferences, I would rank them roughly like this:

1. Insurance Charges
2. Bike Sharing (hourly)
3. Auto MPG
4. Credit
5. Wage
6. Carseats
7. House Sales in King County
8. Diamonds
9. Salaries
10. Forest Fires

These are especially useful because they naturally support:

- immutable vs actionable features,
- monotonic or directional preferences,
- realistic stakeholder narratives,
- easy verbal explanation of priorities.

## Best datasets for easy integration

If you want the least engineering effort first:

1. Diamonds
2. Tips
3. Wage
4. Credit
5. Carseats
6. CPS85
7. Salaries
8. Abalone
9. Auto MPG
10. Insurance Charges

## Easiest download paths

### UCI via `ucimlrepo`

```python
from ucimlrepo import fetch_ucirepo

abalone = fetch_ucirepo(id=1)
auto_mpg = fetch_ucirepo(id=9)
servo = fetch_ucirepo(id=87)
forest_fires = fetch_ucirepo(id=162)
bike_sharing = fetch_ucirepo(id=275)
```

### Built-in or package datasets

```python
import seaborn as sns

diamonds = sns.load_dataset("diamonds")
tips = sns.load_dataset("tips")
```

### `Rdatasets` direct CSVs

```python
import pandas as pd

wage = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Wage.csv")
credit = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Credit.csv")
college = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/College.csv")
carseats = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv")
cps85 = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/CPS85.csv")
salaries = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/carData/Salaries.csv")
hitters = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Hitters.csv")
penguins = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv")
```

### Simple CSV mirror

```python
import pandas as pd

insurance = pd.read_csv(
    "https://www.statlect.com/datasets/SimpleR-Pre-loaded-Medical-insurance-costs.csv"
)
```

## Recommendation for the next selection round

If your main criterion is "good dataset for mathematical work plus easy human
story plus realistic priorities", I would shortlist these first:

1. Insurance Charges
2. Bike Sharing (hourly)
3. Auto MPG
4. Credit
5. Wage
6. Carseats
7. Diamonds
8. House Sales in King County
9. Salaries
10. Abalone

If instead your main criterion is "easiest to integrate quickly", use:

1. Diamonds
2. Wage
3. Credit
4. Carseats
5. CPS85
6. Salaries
7. Abalone
8. Auto MPG
9. Insurance Charges
10. Tips
