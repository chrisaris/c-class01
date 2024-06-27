# c-class01
This is a classification project using census data to determine whether an individual makes over or under $50k/year

# Packages
sklearn
numpy
pandas
matplotlib.pyplot
category_encoders

# Outline
Three models are used, a Naive Bayes Model, Logistic Regression and a Multilayer Perceptron; alternate variants are generated via pipelines.
The initial NB model uses RobustScaler where features are centered and scaled independetly.
Alternative NB models employ either a MinMaxScaler or StandardScaler
The initial Logistic Regression model uses a lbfgs solver with the default l2 penalty
Alternative Logistic Regression models use a liblinear solver or saga solver with l1 and l2 penalties
The initial MLP solver is adam (a stochastic optimizer) with l2 regularization and a constant learning rate
Alternative MLP models use one of either the lbfgs solver with a constant learning rate or a sgd solver with an adaptive learning rate
