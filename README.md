# Medical-Insurance-Cost-Prediction

The project is aimed to predict medical insurance charges using machine learning techniques. The dataset consisted of features such as age, sex, BMI, number of children, smoker status, and region, with the target variable being insurance charges. Categorical variables like sex, smoker status, and region were appropriately encoded for modeling purposes.

To assess the model's performance, the dataset has been divided into training (80%) and test (20%) sets. For predictive analysis, a Linear Regression model has been deployed, which aims to establish a relationship between the aforementioned features and insurance charges.

The model's effectiveness was evaluated using the R-squared metric, which quantifies the proportion of the variance in the target variable that the model explains. The R-squared value obtained on the test dataset was 0.745, indicating that roughly 74.5% of the variance in insurance charges could be explained by the features used in the model.

Several factors might influence insurance charges. For instance, age could correlate with health risks, impacting insurance costs. The BMI might be an indicator of potential health issues, influencing the charges. Smoker status often leads to higher insurance premiums due to associated health risks. Regional differences might also play a role, considering variations in healthcare costs and lifestyles across different areas.

Model's R-squared value of 0.745 suggests that the chosen features capture a significant portion of the variability in insurance charges. Further enhancements could involve exploring additional features or employing more complex models to improve predictive accuracy. Additionally, delving deeper into specific regional or demographic effects could refine the model for more precise predictions.
