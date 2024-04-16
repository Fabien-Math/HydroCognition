import customtkinter as ctk
import tkinter as tk
import os
from PIL import Image
import cv2
from tkvideoplayer import TkinterVideo		# Video preview
import subprocess 							# Enable to process video without freezing the app

import numpy as np
import matplotlib.pyplot as plt

from nn_app_program import load_model, evaluate_model, process_images, resize_images, process_video
from HCRCNN import locate_fish, display_result

# Sets the appearance of the window
## Supported modes : Light, Dark, System
## "System" sets the appearance mode to the appearance mode of the system
ctk.set_appearance_mode("Light") 

# Sets the color of the widgets in the window
# Supported themes : green, dark-blue, blue 
ctk.set_default_color_theme("dark-blue") 

# Dimensions of the window
ratio = 16/9
appWidth = 1600
appHeight = appWidth*9/16

# App Class
class App(ctk.CTk):
	# The layout of the window will be written
	# in the init function itself
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.resizable(width=False, height=False)

		self.current_dir = os.path.dirname(__file__) + '\\'
		self.paths_checkbox = []
		self.process_checkbox = []
		self.hlines = []
		self.paths = []
		self.images = []
		self.images_to_process = []
		self.image_labels = []
		self.model = load_model()
		self.width = 256
		self.height = 144
		self.nb_images = 0
		self.nb_selected_images = 0
		self.nb_selected_process_images = 0
		
		# DEFINE FONTS
		TITLE_FONT = ctk.CTkFont(family="calibri" ,size=20, weight="bold", underline=True)
		SUBTITLE_FONT = ctk.CTkFont(family="calibri", size=16, weight="bold")
		LOG_FONT = ctk.CTkFont(family="calibri", size=14, weight="bold")
		BODY_FONT = ctk.CTkFont(family="calibri", size=14)


		# Sets the title of the window
		self.title("Hydro'Cognition") 
		# Sets the dimensions of the window
		self.geometry(f"{appWidth}x{appHeight}") 

		##### GENERAL FORM #####

		# Vertical separator line
		self.vLine = ctk.CTkFrame(self, width=2, height=900, corner_radius=0, border_width=0, fg_color="gray")
		self.vLine.place(x = appWidth / 2, y = appHeight / 2, anchor='center')

		# Left top horizontal line separator
		self.lhLine_top = ctk.CTkFrame(self, width=1600/2, height=2, corner_radius=0, border_width=0, fg_color="gray")
		self.lhLine_top.place(x = appWidth / 2, y = 230, anchor='e')

		# Left top horizontal line separator
		self.lhLine_bottom = ctk.CTkFrame(self, width=1600/2, height=2, corner_radius=0, border_width=0, fg_color="gray")
		self.lhLine_bottom.place(x = appWidth / 2, y = 650, anchor='e')

		# Left top horizontal line separator
		self.lhLine_bottom = ctk.CTkFrame(self, width=1600/2, height=2, corner_radius=0, border_width=0, fg_color="gray")
		self.lhLine_bottom.place(x = appWidth / 2, y = 530, anchor='e')

		# Left middle horizontal line separator
		self.lhLine = ctk.CTkFrame(self, width=1600/4, height=2, corner_radius=0, border_width=0, fg_color="gray")
		self.lhLine.place(x = 0, y = 130, anchor='w')
		
		# Right horizontal line separator
		self.rhLine = ctk.CTkFrame(self, width=1600/2, height=2, corner_radius=0, border_width=0, fg_color="gray")
		self.rhLine.place(x = appWidth / 2, y = appHeight / 2, anchor='w')



		##### DATASET SECTION #####
		# self.dataset_title = ctk.CTkLabel(self, text="DATASET INFORMATIONS", font=TITLE_FONT)
		# self.dataset_title.place(x = 10, y = 10, anchor='nw')

		# Images width label
		self.width_label = ctk.CTkLabel(self, text="Images width : ", font=BODY_FONT)
		self.width_label.place(x = 10, y = 160, anchor='w')

		# Images width input
		self.width_entry = ctk.CTkEntry(self, height=25, width=60, placeholder_text="256", font=BODY_FONT)
		self.width_entry.place(x = 110, y = 160, anchor='w')

		# Image height label
		self.height_label = ctk.CTkLabel(self, text="Images height : ", font=BODY_FONT)
		self.height_label.place(x = 300, y = 160, anchor='w')

		# Image height input
		self.height_entry = ctk.CTkEntry(self, height=25, width=60, placeholder_text="144", font=BODY_FONT)
		self.height_entry.place(x = 410, y = 160, anchor='w')

		# In app model checkbox
		self.in_app_model_chbox = ctk.CTkCheckBox(self, width=20, height=20, text="In app neural network")
		self.in_app_model_chbox.place(x = 620, y = 200, anchor='w')



		# # Dataset directories label
		# self.input_dir_label = ctk.CTkLabel(self, text="Dataset directories", font=SUBTITLE_FONT)
		# self.input_dir_label.place(x = 10, y = 120, anchor='w')


		# # Create dataset button
		# self.create_ds_button = ctk.CTkButton(self, text="CREATE DATASET", command=self.create_dataset, font=SUBTITLE_FONT)
		# self.create_ds_button.place(x = 700, y = 250, anchor='e')


		##### NEURAL NETWORK #####
		self.nn_title = ctk.CTkLabel(self, text="NEURAL NETWORK SECTION", font=TITLE_FONT)
		self.nn_title.place(x = 10, y = 10, anchor='nw')

		# Model name label
		self.input_dir_label = ctk.CTkLabel(self, text="Model name :", font=SUBTITLE_FONT)
		self.input_dir_label.place(x = 10, y = 60, anchor='w')

		# Model name input
		self.model_name_entry = ctk.CTkEntry(self, height=25, width=100, font=BODY_FONT)
		self.model_name_entry.place(x = 120, y = 60, anchor='w')

		# Model directory label
		self.working_dir_label = ctk.CTkLabel(self, text="Model directory", font=BODY_FONT, wraplength=500, anchor="w")
		self.working_dir_label.place(x = 270, y = 100, anchor='w')

		# Model directory button
		self.working_dir_button = ctk.CTkButton(self, text="Browse model working directory", font=BODY_FONT, command=self.browse_work_folder, width=250)
		self.working_dir_button.place(x = 10, y = 100, anchor='w')


		# Create model button
		self.evaluate_model_button = ctk.CTkButton(self, text="EVALUATE MODEL", command=self.evaluate_model_in_app, font=SUBTITLE_FONT)
		self.evaluate_model_button.place(x = 800/3, y = 200, anchor='center')

		# Load model button
		self.load_model_button = ctk.CTkButton(self, text="LOAD MODEL", command=self.load_model_in_app, font=SUBTITLE_FONT)
		self.load_model_button.place(x = 2*800/3, y = 200, anchor='center')



		##### PROCESS SECTION #####
		
		# Process video title label
		self.video_processing_label = ctk.CTkLabel(self, text="Video processing", font=TITLE_FONT, anchor="w")
		self.video_processing_label.place(x = 10, y = 250, anchor='w')

		# Process video title label
		self.image_processing_label = ctk.CTkLabel(self, text="Image processing", font=TITLE_FONT, anchor="w")
		self.image_processing_label.place(x = 10, y = 550, anchor='w')

		# Video to process directory label
		self.input_dir_video_label = ctk.CTkLabel(self, text="Path to the video selected", font=BODY_FONT, wraplength=350, anchor="w")
		self.input_dir_video_label.place(x = 10, y = 310, anchor='w')

		# Video to process directory button
		self.input_dir_video_button = ctk.CTkButton(self, text="Select video to process", font=BODY_FONT, command=self.browse_input_video_file, width= 200)
		self.input_dir_video_button.place(x = 10, y = 285, anchor='w')

		# Dataset output directory label
		self.output_video_dir_label = ctk.CTkLabel(self, text="Output directory for processed frames", font=BODY_FONT, wraplength=350, anchor="w")
		self.output_video_dir_label.place(x = 10, y = 365, anchor='w')

		# Dataset output directory button
		self.output_video_dir_button = ctk.CTkButton(self, text="Browse output directory", font=BODY_FONT, command=self.browse_video_output_folder, width= 200)
		self.output_video_dir_button.place(x = 10, y = 340, anchor='w')

		# Video preview
		self.video_player = TkinterVideo(master=self, scaled=True)
		self.video_player.place(x=790, y=240, anchor="ne")

		# Preview video button
		self.preview_video_button = ctk.CTkButton(self, text="Preview video", command=self.preview_video, font=SUBTITLE_FONT)
		self.preview_video_button.place(x = 285, y = 285, anchor='w')

		# Pause video button
		self.pause_video_button = ctk.CTkButton(self, text="Play", command=self.play_video, font=SUBTITLE_FONT)
		self.pause_video_button.place(x = 285, y = 320, anchor='w')

		# Process video button
		self.process_video_button = ctk.CTkButton(self, text="PROCESS VIDEO", command=self.process_video_in_app, font=SUBTITLE_FONT)
		self.process_video_button.place(x = 285, y = 470, anchor='w')

		# Process selected images button
		self.process_images_button = ctk.CTkButton(self, width=250, text="PROCESS ALL SELECTED IMAGES", command=self.process_images_in_app, font=SUBTITLE_FONT)
		self.process_images_button.place(x = 750, y = 590, anchor='se')

		# Matplotlib explication label
		self.mat_label = ctk.CTkLabel(self, text="Number of line and column of the result given by the AI in the matplotlib window :", font=BODY_FONT)
		self.mat_label.place(x = 25, y = 600, anchor="nw")

		# Lines input
		self.n_entry = ctk.CTkEntry(self, height=25, width=100, font=BODY_FONT, placeholder_text="Lines")
		self.n_entry.place(x = 500, y = 600, anchor='nw')

		# Columns input
		self.m_entry = ctk.CTkEntry(self, height=25, width=100, font=BODY_FONT, placeholder_text="Columns")
		self.m_entry.place(x = 750, y = 600, anchor='ne')


		##### LOGS #####
  
		# NN log label
		self.log_label = ctk.CTkLabel(self, text="Logs terminal", font=SUBTITLE_FONT)
		self.log_label.place(x = 10, y = 890-200, anchor="sw")

		# NN log textbox
		self.log_textbox = ctk.CTkTextbox(self, width=780, height=200, font=LOG_FONT)
		self.log_textbox.place(x = 10, y = 890, anchor="sw")
		## log textbox configuration 
		self.log_textbox.tag_config('succes', foreground="green")
		self.log_textbox.tag_config('warning', foreground="orange")
		self.log_textbox.tag_config('error', foreground="red")
		self.log_textbox.insert("insert", f"Default model loaded!\n\n", "succes")	

		# NN delete log button
		self.delete_log_button = ctk.CTkButton(self, text="Clear logs", height=25, hover_color='red3', command=self.delete_logs, font=BODY_FONT)
		self.delete_log_button.place(x = 790, y = 890-205, anchor="se")


		##### IMAGES SECTION #####
		## Label
		self.filesPathsLabel = ctk.CTkScrollableFrame(self, width=700, height=300, label_text="Image selector (Preview limited to 4 images)", fg_color=("#DDDDDD", "#111111"))
		self.filesPathsLabel.place(x=3*1600/4, y=200, anchor='center')
		
		## Browse files button
		browse_files_button = ctk.CTkButton(self,
							text = "Browse files",
							command = self.browse_files)
		browse_files_button.place(x=1250, y = 450-50, anchor='center')

		## Select all button
		select_all_process_button = ctk.CTkButton(self, width = 80,
							text = "Select all",
							command = self.select_all_process)
		select_all_process_button.place(x=1550, y = 450-50, anchor='e')

		## Select all button
		deselect_all_process_button = ctk.CTkButton(self, width = 80,
							text = "Deselect all",
							command = self.deselect_all_process)
		deselect_all_process_button.place(x=1450, y = 450-50, anchor='e')

		## Select all button
		select_all_button = ctk.CTkButton(self, width = 80,
							text = "Select all",
							command = self.select_all)
		select_all_button.place(x=850, y = 450-50, anchor='w')

		## Select all button
		deselect_all_button = ctk.CTkButton(self, width = 80,
							text = "Deselect all",
							command = self.deselect_all)
		deselect_all_button.place(x=950, y = 450-50, anchor='w')

		## Delete selected button
		delete_selected_button = ctk.CTkButton(self, width = 80, hover_color="red3",
							text = "Delete selected",
							command = self.delete_selected)
		delete_selected_button.place(x=1050, y = 450-50, anchor='w')

		# Number of image selected label
		self.image_selected_label = ctk.CTkLabel(self, text=f"Selected images : {self.nb_selected_images} / {self.nb_images}")
		self.image_selected_label.place(x = 850, y = 420, anchor='nw')

		# Number of image process selected label
		self.image_process_selected_label = ctk.CTkLabel(self, text=f"Selected process images : {self.nb_selected_process_images} / {self.nb_images}")
		self.image_process_selected_label.place(x = 1550, y = 420, anchor='ne')
		
	##### FUNCTION #####
	## Selector functions
	def browse_files(self):
		"""
		## Description
			Browse files to get all you the one you want
			
		"""
		filenames = ctk.filedialog.askopenfilenames(initialdir = "/", title = "Select a File",
											filetypes = (("Image file", "*.png *.jpg *.jpeg *.gif"), ("all files","*.*")))
		
		self.update_selector(filenames)
		self.update_nb_images()
		self.update_nb_process_images()
		
	def update_selector(self, filenames):
		# Change label contents
		for filename in filenames:
			textToDisplay = ""

			# Extract the last folder and the file
			cpt = 0
			for i in range(len(filename)-1, -1, -1):
				textToDisplay = filename[i] + textToDisplay
				if filename[i] == '/':
					cpt += 1
					if cpt == 2:
						break
			
			# Create checkboxes and vertical separator
			path_chbox = ctk.CTkCheckBox(self.filesPathsLabel, width=20, height=20, text="  "+textToDisplay, command=self.preview_images)
			proc_chbox = ctk.CTkCheckBox(self.filesPathsLabel, width=20, height=20, text="Process image", command=self.update_nb_process_images)
			hLine = ctk.CTkFrame(self.filesPathsLabel, width=1600/2, height=2, corner_radius=0, border_width=0, fg_color=("#CCCCCC", "#222222"))
			self.paths_checkbox.append(path_chbox)
			self.process_checkbox.append(proc_chbox)
			self.hlines.append(hLine)
			self.paths.append(filename)
			path_chbox.pack(anchor="w")
			proc_chbox.pack(pady=2, anchor="e")
			hLine.pack(pady=1, anchor='center')
		
	def select_all_process(self):
		for chk in self.process_checkbox:
			chk.select()
		self.update_nb_process_images()

	def deselect_all_process(self):
		for chk in self.process_checkbox:
			chk.deselect()
		self.update_nb_process_images()

	def select_all(self):
		for chk in self.paths_checkbox:
			chk.select()
		self.update_nb_images()
		self.preview_images()

	def deselect_all(self):
		for chk in self.paths_checkbox:
			chk.deselect()
		self.update_nb_images()
		self.preview_images()

	def delete_selected(self):
		temp_paths = []
		for i, chk in enumerate(self.paths_checkbox):
			if not chk.get():
				temp_paths.append(self.paths[i])
		
		for chk in self.paths_checkbox:
			chk.destroy()
		for chk in self.process_checkbox:
			chk.destroy()
		for line in self.hlines:
			line.destroy()

		self.paths_checkbox = []
		self.process_checkbox = []
		self.paths = []
		self.update_selector(temp_paths)
		self.update_nb_images()
		self.update_nb_process_images()
		self.preview_images()

	def update_nb_images(self):
		self.nb_images = len(self.paths)
		self.nb_selected_images = len([1 for chk in self.paths_checkbox if chk.get()])

		self.image_selected_label.configure(require_redraw=True, text=f"Selected images : {self.nb_selected_images} / {self.nb_images}")

	def update_nb_process_images(self):
		self.nb_images = len(self.paths)
		self.nb_selected_process_images = len([1 for chk in self.process_checkbox if chk.get()])

		self.image_process_selected_label.configure(require_redraw=True, text=f"Selected images : {self.nb_selected_process_images} / {self.nb_images}")

	## Other function	
	def browse_input_img_folder(self):
		dir = ctk.filedialog.askdirectory(initialdir = "/")
		self.input_dir_img_label.configure(require_redraw=True, text=dir)
		
	def browse_input_video_file(self):
		# video_extensions = ('webm', 'mkv', 'flv', 'vob', 'ogv', 'ogg', 'rrc', 'gifv', 'mng', 'mov', 'avi', 'qt', 'wmv', 'yuv', 'rm', 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'flv', 'f4v', 'f4p', 'f4a', 'f4b', 'mod')
		vid_exts = "*.webm *.mkv *.flv *.vob *.ogv *.ogg *.rrc *.gifv *.mng *.mov *.avi *.qt *.wmv *.yuv *.rm *.asf *.amv *.mp4 *.m4p *.m4v *.mpg *.mp2 *.mpeg *.mpe *.mpv *.m4v *.svi *.3gp *.3g2 *.mxf *.roq *.nsv *.flv *.f4v *.f4p *.f4a *.f4b *.mod"
		filename = ctk.filedialog.askopenfilename(initialdir = self.current_dir, title = "Select a File",
											filetypes = (("Video files", vid_exts), ("All files","*.*")))
		self.input_dir_video_label.configure(require_redraw=True, text=filename)
		
	def browse_video_output_folder(self):
		dir = ctk.filedialog.askdirectory(initialdir = "/")
		self.output_video_dir_label.configure(require_redraw=True, text=dir)

	def browse_work_folder(self):
		dir = ctk.filedialog.askdirectory(initialdir = "/")
		self.working_dir_label.configure(require_redraw=True, text=dir)

	def evaluate_model_in_app(self):
		if self.width_entry.get() != "":
			try:
				self.width = float(self.width_entry.get())
			except ValueError:
				self.log_textbox.insert("insert", f"Warning while evaluating the model ! Width is not a valid number !\n", "warning")
				self.log_textbox.insert("insert", f"Width must be an integer between 1 and 1920 ! Set by default to 256.\n\n")
		else:
			self.log_textbox.insert("insert", f"Default width : 256.\n\n")


		if self.height_entry.get() != "":
			try:
				self.height = float(self.height_entry.get())
			except ValueError:
				self.log_textbox.insert("insert", f"Warning while evaluating the model ! Height is not valid number !\n", "warning")
				self.log_textbox.insert("insert", f"Height must be an integer between 1 and 1080 ! Set by default to 144.\n\n")
		else:
			self.log_textbox.insert("insert", f"Default height : 144.\n\n")

		if self.model == None:
			self.log_textbox.insert("insert", f"Error while evaluating the model ! No model loaded !\n", "error")

		
		if os.path.isdir(self.working_dir_label._text) or self.in_app_model_chbox.get():
			if os.path.exists(self.working_dir_label._text) or self.in_app_model_chbox.get():
				h, n = evaluate_model(self.model, self.width, self.height, self.working_dir_label._text)
		else:
			self.log_textbox.insert("insert", f"Error while evaluating the model ! No model found in the directory !\n\n", "error")
			return		
		
		self.log_textbox.insert("insert", f"Number of images in the validation set : {n}\n")
		self.log_textbox.insert("insert", f"Loss: {h[0]:5.3f}\nClassifier accuracy: {h[1]:5.3f}\nRegressor MSE: {h[2]:5.3f}\n\n")

	def load_model_in_app(self):
		if self.in_app_model_chbox.get():
			self.model = load_model()
			if self.model != None:
				self.log_textbox.insert("insert", f"In app model loaded!\n\n", "succes")
			else:
				self.log_textbox.insert("insert", f"Error while loading the in app model!\n\n", "error")	
		else:
			if os.path.isdir(self.working_dir_label._text):
				if os.path.exists(self.working_dir_label._text):
					self.model = load_model(self.model_name_entry.get(), self.working_dir_label._text)
					if self.model == -1:
						self.log_textbox.insert("insert", f"Error! No model found in this directory!\n\n", "error")	
					elif self.model != None:
						self.log_textbox.insert("insert", f"Custom model loaded!\n\n", "succes")
					else:
						self.log_textbox.insert("insert", f"Error while loading the custom model!\n\n", "error")	

			else:
				self.log_textbox.insert("insert", f"Error! No model found in this directory!\n\n", "error")
				return

	def process_images_in_app(self):
		try:
			self.width = float(self.width_entry.get())
		except ValueError:
			self.width_entry.configure(textvariable="256")
			self.width = 256
		try:
			self.height = float(self.height_entry.get())
		except ValueError:
			self.height_entry.configure(textvariable="144")
			self.height = 144

		self.images_to_process = []
		paths = []
		for i, checkBox in enumerate(self.process_checkbox):
			if checkBox.get():
				image_path = self.paths[i]	
				paths.append(self.paths[i])
				self.images_to_process.append(cv2.imread(image_path))

		# self.images_to_process = resize_images(self.images_to_process, self.width, self.height)
		# self.images_to_process = np.array(self.images_to_process)

		for i, image in enumerate(self.images_to_process):
			img_height, img_width, _ = np.shape(image)
			interest_matrix, predictions = locate_fish(self.model, image)
			
			if predictions[-1] < 0.5:
				self.log_textbox.insert("insert", f"No fish in {paths[i]}\n\n")
			else:
				plt.subplot(int(self.n_entry.get()), int(self.m_entry.get()), i+1)
				display_result(image, interest_matrix, img_width, img_height)

		plt.show()

		# self.show_images(self.images_to_process, labels, int(self.n_entry.get()), int(self.m_entry.get()))

		self.log_textbox.insert("insert", f"Images processed\n\n", "succes")

	def process_video_in_app(self):
		self.log_textbox.insert("insert", "Processing video...")
		
		process_video(self.model, self.input_dir_video_label._text, 256, 144, self.output_video_dir_label._text)

		self.log_textbox.insert("insert", "Video processed successfully !", "success")

	def show_images(self, train_images, train_labels, N = 4, M = 4):
		plt.figure("Results")
		for i in range(min(N*M, len(train_images))):
			image, label = train_images[i], train_labels[i]

			image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			text = "no_fish"
			if label == 1:
				text = "fish"

			plt.subplot(N, M, i + 1)
			plt.imshow(image_color)
			plt.title(text, fontsize=8)
			plt.axis("off")
		
		plt.show()

	def preview_images(self):
		"""
		## Description
			Show images in the preview frame
			
		"""

		# Destroy old images and labels
		for img in self.images:
			img.destroy()
		for text in self.image_labels:
			text.destroy()

		# Count how many checkboc are check
		cpt = 0
		for i, checkBox in enumerate(self.paths_checkbox):
			if checkBox.get():
				cpt += 1
		
		# Define image sizes
		img_sizes = {
			'1': [(700, 350)],
			'2': ((700, 170), (700, 170)),
			'3': ((700, 170), (345, 170), (345, 170)),
			'4': ((345, 170), (345, 170), (345, 170), (345, 170))
		}

		# Define image placements
		img_places = {
				'1': [{"x": 1200, "y": 450 + 50 + 350/2}],
			   '2': ({"x": 1200, "y": 450 + 50 + 170/2},
					{"x": 1200, "y":900 - 50 - 170/2}),
			   '3': ({"x": 1200, "y": 450 + 50 + 170/2},
					{"x": 800 + 50 + 345 / 2, "y": 900 - 50 - 170/2},
					{"x": 1600 - 50 - 345/2, "y": 900 - 50 - 170/2}),
			   '4': ({"x": 800 + 50 + 345 / 2, "y": 450 + 50 + 170/2},
					{"x": 1600 - 50 - 345/2, "y": 450 + 50 + 170/2},
					{"x": 800 + 50 + 345 / 2, "y": 900 - 50 - 170/2},
					{"x": 1600 - 50 - 345/2, "y": 900 - 50 - 170/2})
					}
		
		# Define label placement
		img_label_places = {
			'1': [{"x": 3*1600/4, "y": 450 + 35}],
			'2': ({"x": 3*1600/4, "y": 450 + 35},
		 		{"x": 3*1600/4, "y": 900 - 35}),
			'3': ({"x": 850 + 700/2, "y":450+35},
		 		{"x": 850 + 345/2, "y": 900 - 35},
				{"x": 850 + 345 + 10 + 345/2, "y": 900 - 35}),
			'4': ({"x": 850 + 345/2, "y": 450 + 35},
		 		{"x": 850 + 345 + 10 + 345/2, "y": 450 + 35},
				{"x": 850 + 345/2, "y": 900 - 35},
				{"x": 850 + 345 + 10 + 345/2, "y": 900 - 35})
				}

		# Show images to be shown
		id_place = 0
		for i, checkBox in enumerate(self.paths_checkbox):
			if checkBox.get():
				nb_image_to_display = str(min(4, cpt))
				image_path = self.paths[i]

				image = Image.open(image_path)
				img_ratio = image.width/image.height
				max_width, max_height = img_sizes[nb_image_to_display][id_place]
				max_ratio = max_width / max_height
				if img_ratio > max_ratio:
					img_size = (max_width, max_width/img_ratio)
				else:
					img_size = (max_height*img_ratio, max_height)

				image = ctk.CTkImage(light_image=Image.open(image_path), size=img_size)

				img_preview = ctk.CTkLabel(self, image=image, text='')
				img_preview.place(x = img_places[nb_image_to_display][id_place]["x"], y = img_places[nb_image_to_display][id_place]["y"], anchor='center')
				
				imgText = ""
				for i in range(len(image_path)-1, -1, -1):
					if image_path[i] == '/':
						break
					imgText = image_path[i] + imgText
				img_label = ctk.CTkLabel(self, text=imgText)
				img_label.place(x = img_label_places[nb_image_to_display][id_place]["x"], y = img_label_places[nb_image_to_display][id_place]["y"], anchor='center')
				
				self.images.append(img_preview)
				self.image_labels.append(img_label)
				id_place += 1

			if id_place == 4:
				break
		
		self.update_nb_images()
		self.update_nb_process_images()

	def preview_video(self):
		try:
			self.video_player.load(self.input_dir_video_label._text)
		except:
			self.log_textbox.insert("insert", "Error while loading the video to render a previiew !", "error")

		self.video_player.set_size((356, 200), keep_aspect=True)

		self.video_player.play()
		self.video_player.pause()

		self.preview_video_button.configure(require_redraw=True, command=self.stop_video_preview, text="Stop preview")

	def pause_video(self):
		self.video_player.pause()
		self.pause_video_button.configure(require_redraw=True, text="Play", command=self.play_video)
	
	def play_video(self):
		self.video_player.play()
		self.pause_video_button.configure(require_redraw=True, text="Pause", command=self.pause_video)

	def stop_video_preview(self):
		self.video_player.stop()
		self.preview_video_button.configure(require_redraw=True, command=self.preview_video, text="Preview video")

	def delete_logs(self):
		self.log_textbox.delete('1.0', 'end')			


if __name__ == "__main__":
	app = App()
	# Used to run the application
	app.mainloop()	 
