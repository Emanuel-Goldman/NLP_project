import os


def name_file(book_name,chap_num,year):
    return book_name + " " + str(chap_num) + " " + str(year) + ".txt"


def text_to_chapter_list(book):
    chapters_list = book.split("CHAPTER ")
    return chapters_list


def get_chap_list(path):
    folder = r"C:\Users\Lenovo\OneDrive - post.bgu.ac.il\שולחן העבודה\Charles Dickens Books\Books"
    books_names = get_books_names(folder)

    for book_name in books_names:

        chap_list = []
        year = get_year(book_name)
        print("the book name is: " + book_name + " and the year is: " + year)

        book_path = folder + "\\" + book_name + ".txt"

        with open(book_path, "r", encoding="utf-8") as file:
            book_text = file.read()

        clean_book_text = clean_book(book_text)

        chapters = text_to_chapter_list(clean_book_text)

        for chapter in chapters:
            # create a file for each chapter
            chap_list.append([chapter, year])
    return chap_list


def clean_book(book_text):
    start = find_position_chap1(book_text)+8
    end = find_position_last_chap(book_text)
    clean_book_text = book_text[start:end]
    return clean_book_text


def get_year(book_name):
    year_string = ""
    for char in book_name:
        if char.isdigit():
            year_string += char
    return year_string


def get_books_names(folder):
    books_names = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            file_name = file.split(".")[0]
            books_names.append(file_name)
    return books_names


def find_position_chap1(text):
    position = text.find("(start)")
    if position == -1:
        position = text.find("CHAPTER I.\n")
    if position == -1:
        position = text.find("Chapter I.\n")
    if position == -1:
        position = text.find("CHAPTER 1. I AM BORN\n")
    return position


def find_position_last_chap(text):
    position = text.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    return position


def main():

    folder = r"C:\Users\Lenovo\OneDrive - post.bgu.ac.il\שולחן העבודה\Charles Dickens Books\Books"
    books_names = get_books_names(folder)
   
    for book_name in books_names:

        year = get_year(book_name)
        print("the book name is: " + book_name + " and the year is: " + year)

        book_path = folder + "\\" + book_name + ".txt"

        with open(book_path, "r", encoding="utf-8") as file:
            book_text = file.read()

        clean_book_text = clean_book(book_text)

        chapters = text_to_chapter_list(clean_book_text)

        for chapter in chapters:
            #create a file for each chapter
            chap_num = chapters.index(chapter)
            chaps_path = r"C:\Users\Lenovo\PycharmProjects\NLP_project\chaps"
            chap_path = chaps_path + "\\" + book_name + " " + str(chap_num) + ".txt"
            with open(chap_path, "w", encoding="utf-8") as file:
                file.write(chapter)

# counting number of chapters
    count = 0
    for file in os.listdir(chaps_path):
        if file.endswith(".txt"):
            count += 1

    print("number of chapters is: " + str(count))
    return


if __name__ == "__main__": 
    main()
