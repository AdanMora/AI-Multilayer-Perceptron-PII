from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from Controller import Controller
from pygame import mixer # Load the required library

class GUI(object):

    def __init__(self):
        self.root = Tk()
        self.controller = Controller()
        self.images_extensions = (("Image files", ("*.jpg","*.png")),)
        self.guitar = "guitar.mp3"

        # Label del titulo
        #self.title = Label(self.root,text="Seleccione el archivo pickle para cargarlo")
        #self.title.grid(row=0,column=0)

        # Boton de elegir pickle
        
        self.choose = Button(self.root,text="Browse",command=self.load_file,width=30)
        self.choose.grid(row=1,column=0)

        # Mensaje de seleccionado
        self.choose_text = "Seleccione un archivo binario"
        self.choose_error = "Eso no es un archivo binario"
        self.choose_msg = Label(self.root,text=self.choose_text)
        self.choose_msg.grid(row=1,column=1)

        # Boton de clasificar
        self.classify = Button(self.root,text="Seleccionar imágen y clasificar",command=self.load_image,width=30)
        self.classify.grid(row=2,column=0)


        # Mensaje de clasificado
        self.classify_text = "Seleccione una imágen"
        self.classify_wait = "Clasificando imágen..."
        self.classify_msg = Label(self.root,text=self.classify_text)
        self.classify_msg.grid(row=2,column=1)

        #self.setup()
        self.root.mainloop()

    def classification(self,image):
        return self.controller.classify(image)

    def set_file(self,file):
        return self.controller.set_file(file)
        
    def load_file(self):
        file = askopenfilename(filetypes=(("Binary files", "*.*"),))

        if file:
            extension = file.split("/")[-1]
            if not "." in extension:
                # selecciono un binary
                # todo
                text = self.set_file(file)
                self.choose_msg.config(text=text)
            else:
                # selecciono uno con extension
                self.choose_msg.config(text=self.choose_error)
        else:
            self.choose_msg.config(text=self.choose_error)



    def load_image(self):
        image = askopenfilename(filetypes=self.images_extensions)

        if image:
            # selecciono una img
            # todo
            self.classify_msg.config(text=self.classify_wait)
            text = self.classification(image)
            self.classify_msg.config(text=text)
            self.play(self.guitar)
        else:
            # selecciono uno con extension
            self.classify_msg.config(text=self.classify_text)

    def play(self,sound):
        mixer.init()
        mixer.music.load(sound)
        mixer.music.play()

GUI()
