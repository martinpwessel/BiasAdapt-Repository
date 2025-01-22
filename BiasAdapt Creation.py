import os
from openai import OpenAI
import ast
import numpy as np # linear algebra
import pandas as pd
import random
from tqdm import tqdm

# Set your OpenAI API key

key = XXX

client = OpenAI(api_key=key)


class DataLoad:
  """
  Class to read in the BABE Dataset
  """

  @staticmethod
  def read_babe():
    df = pd.read_excel("final_labels_SG2.xlsx")
    lst = []
    for index, row in df.iterrows():
      if row['label_bias'] == "No agreement":
        pass
      else:
        sub_dict = {'text': row['text']}
        if row['label_bias'] == "Biased":
          sub_dict['label'] = 1
        elif row['label_bias'] == "Non-biased":
          sub_dict['label'] = 0
        lst.append(sub_dict)
    return lst


data = DataLoad.read_babe()


class AdverserialExamples:
  def __init__(self, data, x, y):
    self.data = data[x:y]
    self.x = x

  def main(self):

    alternatives = self.generate_alternatives()
    self.save_intermediate(alternatives, 'alternatives_' + str(self.x) + '.xlsx', True)
    results = []
    for i, alternative in enumerate(alternatives):
      original = self.data[i]
      results.append(self.generate_alternative_texts(original, [alternative[1]]))
    self.save_to_excel(results, 'output_' + str(self.x) + '.xlsx')

  def identify_words(self):
    responses = []

    for i, entry in enumerate(tqdm(self.data, desc="Processing", unit="entry")):
      try:
        response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {
              "role": "system",
              "content": """You will be provided with a text. Does this text contain words associated with gender, origin, disability, religion, political affiliation, occupation, or the name of a politician? 
              Please return a list containing only the words associated only with gender, origin, disability, religion, political affiliation, occupation, or the name of a politician in a complete Python dictionary of the form {"word1": category,"word2":category,...}. 
              If you don’t find any associated words, output None instead of the Dictionary.
              Only output this. Do not output any other text, such as an explanation. """
            },
            {
              "role": "user",
              "content": entry['text']
            }
          ],
          temperature=0.4,
          max_tokens=150,
          top_p=0.5
        )

        res = response.choices[0].message.content
        responses.append([i, ast.literal_eval(res)])
      except SyntaxError:
        print('IDENTIFY WORDS ERROR', i, res)
        responses.append([i, None])
    self.save_intermediate(responses, 'identified_words_' + str(self.x) + '.xlsx', False)
    return responses

  def generate_alternatives(self):
    responses = self.identify_words()
    alternatives = []
    for i, lst in enumerate(tqdm(responses, desc="Processing", unit="entry")):
      try:
        response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {
              "role": "system",
              "content": """You are provided with a dictionary. The keys of the dictionary are words that you should find alternatives for, the entries are the associated categories.
              Find three alternatives for each of the words based on the following rules:
              gender - find alternative words that are associated with the opposite gender and gender-neutral.
              origin - find alternative words that are associated with origins from other continents.
              disability - find alternative words that are associated with other disabilities.
              religion - find alternative words that are associated with other religions.
              political affiliation - find alternative words that are associated with other political affiliations.
              occupation - find alternative words that are associated with other occupations.
              name of a politician - find alternative politician names that are associated with other political affiliations.
              Output in the following form: {"word": ["alternative", "alternative", "alternative"], "word": ["alternative", "alternative", "alternative"],...}
              Only output this. Do not output any other text, such as an explanation. """
            },
            {
              "role": "user",
              "content": str(lst[1])
            }
          ],
          temperature=0.4,
          max_tokens=250,
          top_p=0.5
        )
        res = response.choices[0].message.content
        alternatives.append([i, ast.literal_eval(res)])
      except SyntaxError:
        print("ALTERNATIVE WORDS ERROR", i, res)
        alternatives.append([i, None])
    return alternatives

  def generate_alternative_texts(self, original, alternatives):
    try:
      text = original['text']
      label = original['label']
      alternative_texts = []

      for _ in range(3):  # Generate three alternative texts
        new_text = text
        for alt in alternatives:
          for key, values in alt.items():
            if key in new_text:
              # Choose a random replacement
              replacement = random.choice(values)
              # Replace all occurrences of the key
              new_text = new_text.replace(key, replacement)

        alternative_texts.append(new_text)
      return {
        'original_text': text,
        'label': label,
        'alternative_texts': alternative_texts}

    except AttributeError:
      print('ALTERNATIVE TEXT ERROR', original, alternatives)
      return {
        'original_text': text,
        'label': label,
        'alternative_texts': None}

  def save_to_excel(self, data_list, excel_filename):
    # Create a list to store data
    data_rows = []

    # Populate the list with data from the input list
    for item in data_list:
      # Add the original text
      data_rows.append({'original': 1, 'text': item['original_text'], 'label': item['label']})

      # Add alternative texts
      try:
        for alt_text in item['alternative_texts']:
          data_rows.append({'original': 0, 'text': alt_text, 'label': item['label']})
      except TypeError:
        print('TypeError, no alternative text found')

    # Convert the list to a DataFrame
    df = pd.DataFrame(data_rows, columns=['original', 'text', 'label'])

    # Save the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)

  def save_intermediate(self, data_list, excel_filename, res_var):

    data_rows = []

    # Populate the list with data from the input list
    for item in data_list:
      index, alternatives_dict = item
      if res_var == True:
        try:
          for key, values in alternatives_dict.items():
            for value in values:
              data_rows.append([index, key, value])
        except AttributeError:
          print('AttributeError, no alternative text found', index, item)
      elif res_var == False:
        try:
          for key, value in alternatives_dict.items():
            data_rows.append([index, key, value])
        except AttributeError:
          print('AttributeError, no alternative text found', index, item)

    # Convert the list to a DataFrame
    df = pd.DataFrame(data_rows, columns=['Index', 'Words', 'Alternatives'])

    # Save the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)


for x in range(0,1000,100):
    y = x+100
    print(x,y)
    alt = AdverserialExamples(data, x, y)
    result = alt.main()
