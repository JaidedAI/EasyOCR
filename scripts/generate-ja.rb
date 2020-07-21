# frozen_string_literal: true

require 'json'
require 'nokogiri'
require 'parallel'
require 'ruby-progressbar'

JMDICT_XML = 'JMdict_e'
JMNEDICT_XML = 'JMnedict.xml'
PUNC = '【】《》〈〉｟｠｛｝［］〔〕（）『』「」、；：・？〜＝。！⁉︎‥…〜※＊〽♪♫♬♩〇〒〶〠〄ⓍⓁⓎ→'.chars

def download_dict(xml)
  return if File.exist?(File.expand_path(xml, __dir__))

  archive = "#{xml}.gz"
  url = "http://ftp.monash.edu/pub/nihongo/#{archive}"
  `cd #{File.dirname(__FILE__)} && wget #{url} && gunzip #{archive}`
end

def read_word(word)
  word.css('k_ele keb').map(&:text) + word.css('r_ele reb').map(&:text)
end

def read_dict(filename, root)
  xml = Nokogiri::XML(File.open(File.expand_path(filename, __dir__)))
  words = xml.css("#{root} > entry")
  Parallel.flat_map(words, in_threads: 16, progress: root) do |word|
    read_word(word)
  end
end

def write_files(words)
  src_dir = File.expand_path('../easyocr', __dir__)
  ja_dict = File.join(src_dir, 'dict', 'ja.txt')
  ja_char = File.join(src_dir, 'character', 'ja_char2.txt')
  ja_char_old = File.join(src_dir, 'character', 'ja_char.txt')
  ja_punc = File.join(src_dir, 'character', 'ja_punc.txt')

  words -= PUNC
  chars = words.join.chars.uniq
  chars_old = IO.read(ja_char_old).split("\n")

  puts "new characters: #{(chars - chars_old).size}"
  puts "missing characters: #{(chars_old - chars).size}"
  puts chars_old - chars

  IO.write(ja_dict, words.join("\n"))
  IO.write(ja_char, chars.join("\n"))
  IO.write(ja_punc, PUNC.join("\n"))
end

download_dict(JMDICT_XML)
download_dict(JMNEDICT_XML)
words = read_dict(JMDICT_XML, 'JMdict') + read_dict(JMNEDICT_XML, 'JMnedict')
write_files(words)
