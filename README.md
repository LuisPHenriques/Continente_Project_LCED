# Continente Project - LCED

## Feature Engineering

    All the variables are time framed

### Explicit features

- **day** - day of the month;
- **week** -  number of the week (1-52);
- **dow** - day of the week (0-6);
- **month** -  month of the year (1-12); 
- **quarter** - quarter of the year (1-4); 
- **year** - calendar year (20**);
- **location** - name of the city, zip code, etc.

### Customer features

- **total number of purchases** - total number of times a customer made a purchase;
- **unique number of categories** bought - distinct number of categories bought by a customer;
- **total number of prior orders** - number of transations made prior to the current transaction;
- **unique number of orders** - number of distinct products/category? combinations per ticket; 
- **avg days since prior transaction** - average number of days since a customer made a purchase;
- **avg basket size** - average number of categories in each transaction;
- **last basket size** - number of categories in the last transaction; 
- **number of categories only bought once** - number of categories bought only once by a customer;
- **reordered categories by a customer** - number of categories reordered by a customer (derived feature: 2ª-8ª);

		mode; 
		1st quartil; 
		median;
		3rd quartil.

### Category features

- **total number of categories bought** - total number of times each category was purchased;
- **number of customers who bought a category** - total number of customers (distinct) who of bought a category;

		mode; 
		1st quartil;
		median 3rd;
		quartil of dow for a category.
    
### Customer-Category features

- **customers who bought only one time** - if a customer bought only one time from a category - 1, otherwise - 0;
- **number of reorders of a customer for a category** - total number of times each category was reordered by a customer; 
- **avg days since a customer bought from a category** - average number of days since the last transaction that included that category; 
- **days since a customer bought from a category** - number of days since the last transaction that included that category;


		mode; 
		1st quartil; 
		median;
		3rd quartil of dow of a customer for a category.
    
    
## 2nd part

Apply feature selection before tensor flow and compare the results.
- Filter methods: Correlations, VIF

Data transformation: 
- min-max normalization vs standardization (normal distribution approach)
- Log transformation

Label: 
- 1 if a customer bought a category in the last x months else 0

Training-validation-test split

Training models, with feature selection (tensorflow evaluates importance of features for predicting the target variable) and hyperparameter tuning. 
- Check if features with high computational cost actually improve well the model’s predictability power.
