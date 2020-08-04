
- 4 August 2020 - Version 1.1.6
    - New language support for Russian, Serbian, Belarusian, Bulgarian, Mongolian, Ukranian (Cyrillic Script) and Arabic, Persian(Farsi), Urdu, Uyghur (Arabic Script)
    - Docker file and Ainize demo (thanks @ghandic and @Wook-2)
    - Better production friendly with Logger and custom model folder location (By setting ` model_storage_directory` when create `Reader` instance) (thanks @jpotter)
    - Model files are now downloaded from github's releases
    - readtext can now accept grayscale image

- 24 July 2020 - Version 1.1.5
    - New language support for Hindi, Marathi, Nepali (Devanagari Script)
    - Automatic word merging into paragraph (Use this feature by setting `readtext`'s parameter `'paragraph' = True`)
