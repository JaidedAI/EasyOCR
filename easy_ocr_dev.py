import easyocr

reader = easyocr.Reader(['fr'])
# reader = easyocr.Reader(['en'])
results = reader.readtext('examples/french.jpg',
                         detail=1,
                         decoder="beamsearch"
                         )

for result in results:
    print(result)
