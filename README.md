## Keras AI to get news categories based on title

Initially used BBC news dataset to train and predict news by titile of 
5 main categories: 

business, sport, tech, politics, entertainment

### Dataset link: 
https://www.kaggle.com/datasets/gpreda/bbc-news

##Main logic: 
- Create categorizer object
- Train model based on dataset path, delimiter in dataset and path to save model dump
- After training, you can use load function with dump path to load existing one
- Use function prepare_text_ro_predict to prepare text

### How does the prediction work: 

To get predicted array:
```
prepared_text = categorizer.prepare_text_ro_predict(text)
predicted = model.predict(prepared_text)[0]
```

To get predict with category names: 
```
category_probabilities = [f"{cat} ({prob:.2%})" for cat, prob in zip(category_list, predicted)]
formatted_output = ", ".join(category_probabilities)
```
