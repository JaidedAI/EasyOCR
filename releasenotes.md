
- 23 August 2020 - Version 1.1.8
    - 20 new language supports for Bengali, Assamese, Abaza, Adyghe, Kabardian, Avar,
    Dargwa, Ingush, Chechen, Lak, Lezgian, Tabassaran, Bihari, Maithili, Angika,
    Bhojpuri, Magahi, Nagpuri, Newari, Goan Konkani
    - Support RGBA input format
    - Add `min_size` argument for `readtext`: for filtering out small text box
- 10 August 2020 - Version 1.1.7
    - New language support for Tamil
    - Temporary fix for memory leakage on CPU mode
- 4 August 2020 - Version 1.1.6
    - New language support for Russian, Serbian, Belarusian, Bulgarian, Mongolian, Ukranian (Cyrillic Script) and Arabic, Persian(Farsi), Urdu, Uyghur (Arabic Script)
    - Docker file and Ainize demo (thanks @ghandic and @Wook-2)
    - Better production friendly with Logger and custom model folder location (By setting ` model_storage_directory` when create `Reader` instance) (thanks @jpotter)
    - Model files are now downloaded from github's releases
    - readtext can now accept grayscale image
- 24 July 2020 - Version 1.1.5
    - New language support for Hindi, Marathi, Nepali (Devanagari Script)
    - Automatic word merging into paragraph (Use this feature by setting `readtext`'s parameter `'paragraph' = True`)
