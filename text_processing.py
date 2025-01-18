#!/usr/bin/env python
# coding: utf-8

# In[19]:


def remove_punc(input_text):
    punctuation_marks = ['.',',','?','#','$','!']
    output_text =""
    for char in input_text :
        if char not in punctuation_marks:
            output_text += char
    return output_text


# In[21]:


remove_punc('''hello!, "how are you?"''')


# In[61]:


def remove_stopwords(input_text):
    stop_words = ["a","an","is","and","The","a","an","in","on"]
    words = input_text.split()
    filtered_words =[]
    for word in words:
        if word.lower() not in stop_words:
            filtered_words.append(word)
    output_text = ' '.join(filtered_words)
    return (output_text)


# In[63]:


remove_stopwords("The movie was good and it is a hit")


# In[ ]:




