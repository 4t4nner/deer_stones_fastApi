import cv2
import json
import numpy as np
# from scipy.ndimage import convolve
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from typing import Optional, Dict, Any

app = FastAPI()

class ImageProcessor:
    """
    Класс для обработки изображений с настраиваемыми параметрами фильтрации и морфологических операций
    
    Параметры инициализации:
        output_path (string): mag/saved.png
        search_contours (boolean): искать контуры?
        img_path (string): defaultCorona_AO.tga
        img_plus_img_size (int): bin_img + processed
        blur_ksize (int): Размер ядра Гауссова размытия (нечётный, 0 - авто)
        sobel_ksize (int): Размер ядра фильтра Собеля (1,3,5,7)
        sharp_k (float): Коэффициент усиления резкости
        alpha (float): Коэффициент контраста (1.0-3.0)
        beta (float): Сдвиг яркости
        gamma (float): Гамма-коррекция
        binary_thresh (int): Порог бинаризации (0-255)
        morph_ops (dict): Словарь морфологических операций {operation: kernel_size}
    """
    
    def __init__(self, 
                 output_path='mag/saved.png',
                 img_path='mag/defaultCorona_AO.tga',
                 search_contours=True,
                 img_plus_img_size=3,
                 blur_ksize=15,
                 sobel_ksize=7,
                 sharp_k=1.5,
                 alpha=9,
                 beta=0,
                 gamma=2.0,
                 binary_thresh=26,
                 morph_ops={'open': 3, 'close': 3}):
        
        # Основные параметры обработки
        self.search_contours = search_contours
        self.output_path = output_path
        self.img_path = img_path
        self.img_plus_img_size = img_plus_img_size
        self.blur_ksize = blur_ksize if blur_ksize%2==1 else blur_ksize+1
        self.sobel_ksize = max(1, min(7, sobel_ksize))
        self.sharp_k = sharp_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.binary_thresh = binary_thresh
        
        # Обработка морфологических операций
        self.morph_ops = {}
        for op, size in morph_ops.items():
            if size > 0:
                self.morph_ops[op] = size if size%2==1 else size+1
    
    def find_contours(self, processed_image):
        """Находит контуры на обработанном изображении"""
        if processed_image is None:
            return None
            
        # Создаём основное изображение и overlay слой
        if len(processed_image.shape) == 2:
            base_img = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        else:
            base_img = processed_image.copy()
            
        overlay = base_img.copy()
        output = base_img.copy()
        
        # Получаем бинарное изображение для поиска контуров
        if len(processed_image.shape) == 3:
            binary = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            binary = processed_image.copy()
            
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_CCOMP, 
            cv2.CHAIN_APPROX_TC89_L1
        )
        
        if not contours:
            return base_img

        # Фильтрация контуров (оставить только крупные)
        contour_areas = []
        for i, cnt in enumerate(contours):
            cntArea = cv2.contourArea(cnt)
            if cntArea > 1000:
                contour_areas.append((i, cntArea))
        contour_areas.sort(key=lambda x: x[1], reverse=True)

        # Удаляем самый большой контур (первый в списке)
        selected_contours = contour_areas[1:]

        selected_indices = [idx for idx, _ in selected_contours]

        # Закрашиваем область внутри контуров
        for idx in selected_indices:
            cv2.fillPoly(overlay, [contours[idx]], color=(0, 0, 255))  # Красный цвет в BGR

        # Смешиваем слои с прозрачностью
        alpha = 0.3  # Уровень прозрачности (30%)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # self.contours_image = output
        return output
    
    def get_parameters_from_console(self):
        """
        Ввод параметров из консоли в формате JSON
        
        Возвращает:
            bool: True если параметры были введены и установлены, False если ввод отменен
        """
        print("\nВведите параметры в JSON формате (или оставьте пустым для ввода по-умолчанию):")
        print("Доступные параметры: blur_ksize, sobel_ksize, sharp_k, alpha, beta, gamma, binary_thresh, morph_ops")
        print("Пример: {'blur_ksize':5, 'sobel_ksize':3, 'morph_ops':{'open':3, 'close':5}}")
        
        json_str = input("> ")
        if not json_str:
            return False
            
        return self.set_parameters_from_json(json_str)

    def interactive_processing_loop(self):
        """
        Бесконечный цикл интерактивной обработки:
        1. Запрашивает параметры через консоль
        2. Обрабатывает изображение
        3. Сохраняет результат
        """
        while True:
            print("\n" + "="*50)
            print("Текущие параметры:")
            print(f"blur_ksize: {self.blur_ksize}")
            print(f"img_path: {self.img_path}")
            print(f"sobel_ksize: {self.sobel_ksize}")
            print(f"morph_ops: {self.morph_ops}")
            print("="*50)
            
            if not self.get_parameters_from_console():
                print("Продолжаю с уже заданными параметрами")
                
            # Загрузка и обработка изображения
            input_path = input(f"Введите путь к входному изображению({self.output_path}): ")
            try:
                if input_path is not None and input_path != '':
                    self.img_path = input_path
                
                pil_image = Image.open(self.img_path)
                self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
                
                if self.original_image is None:
                    raise FileNotFoundError
                processed = self.process_image(self.original_image)
                
                output_path = input(f"Введите путь для сохранения результата (enter для пути '{self.output_path}'): ")
                if output_path != '':
                    self.output_path = output_path
                self.save_image(processed, self.output_path)
                print(f"Изображение успешно сохранено в {self.output_path}")
                
            except Exception as e:
                print(f"Ошибка обработки: {e}")

    def set_parameters_from_json(self, json_str):
        """
        Установка параметров из JSON строки
        
        Параметры:
            json_str (str): JSON строка с параметрами или None/пустая строка
            
        Возвращает:
            bool: True если параметры были установлены, False если строка пустая
        """
        if not json_str or json_str.strip() == '':
            return False
            
        try:
            params = json.loads(json_str)
            self.set_parameters(**params)
            return True
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            return False

    def set_parameters(self, **kwargs):
        """
        Обновление параметров обработки
        
        Допустимые параметры:
            blur_ksize (int), sobel_ksize (int), sharp_k (float)
            alpha (float), beta (float), gamma (float)
            binary_thresh (int), morph_ops (dict)
        """
        if 'morph_ops' in kwargs:
            self.morph_ops = {}
            for op, size in kwargs['morph_ops'].items():
                if size > 0:
                    self.morph_ops[op] = size if size%2==1 else size+1
            kwargs.pop('morph_ops')
            
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key in ['blur_ksize', 'sobel_ksize']:
                    value = value if value%2==1 else value+1
                setattr(self, key, value)

    def _apply_morph_ops(self, image):
        """Применение морфологических операций"""
        kernel_types = {
            'dilate': cv2.MORPH_DILATE,
            'erode': cv2.MORPH_ERODE,
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE
        }
        processed = image.copy()
        
        for op, size in self.morph_ops.items():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            processed = cv2.morphologyEx(processed, kernel_types[op], kernel)
            
        return processed
    
    def _imgPlusImg(self, bin_img, processed_img, size):
        kernelEl3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        processed_img = processed_img - cv2.morphologyEx(processed_img, cv2.MORPH_DILATE, kernelEl3)
        return bin_img + processed_img

    def _enhance_image(self, image):
        """Применение гамма-коррекции и контрастирования"""
        look_up_table = np.array([((i / 255.0) ** self.gamma) * 255 
                                for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(image, look_up_table)
        return cv2.convertScaleAbs(enhanced, alpha=self.alpha, beta=self.beta)

    def process_image(self, image):
        """
        Основной метод обработки изображения
        
        Параметры:
            image (numpy.ndarray): Входное изображение в формате BGR
            
        Возвращает:
            numpy.ndarray: Обработанное бинарное изображение
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Гауссово размытие
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        
        # Увеличение резкости
        sharpened = cv2.addWeighted(gray, 1.0 + self.sharp_k, blurred, -self.sharp_k, 0)
        
        # Фильтр Собеля
        grad_x = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        grad_y = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        edges = np.sqrt(grad_x**2 + grad_y**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)
        
        # Улучшение и бинаризация
        enhanced = self._enhance_image(edges)
        _, binary = cv2.threshold(enhanced, self.binary_thresh, 255, cv2.THRESH_BINARY)
        
        bin_img_copy = binary.copy()
        
        # Морфологические операции
        processed = self._apply_morph_ops(binary)

        if (self.img_plus_img_size):
            processed = self._imgPlusImg(bin_img_copy, processed, self.img_plus_img_size)

        if (self.search_contours):
            processed = self.find_contours(processed)

        return processed

    def save_image(self, image, filename):
        """
        Сохранение изображения
        
        Параметры:
            image (numpy.ndarray): Обрабатываемое изображение
            filename (str): Путь для сохранения
        """
        cv2.imwrite(filename, image)

    def set_from_console(self):
        """Интерактивная установка параметров через консоль"""
        params = {
            'img_plus_img_size': int(input("bin_img + processed: ")),
            'blur_ksize': int(input("Gaussian blur kernel size: ")),
            'sobel_ksize': int(input("Sobel kernel size (1,3,5,7): ")),
            'sharp_k': float(input("Sharpness coefficient: ")),
            'alpha': float(input("Contrast alpha: ")),
            'beta': float(input("Brightness beta: ")),
            'gamma': float(input("Gamma correction: ")),
            'binary_thresh': int(input("Binary threshold (0-255): "))
        }
        self.set_parameters(**params)

# Создаем экземпляр процессора для использования в API
processor = ImageProcessor()

@app.post('/api/process_image')
async def process_image_api(
    image_path: str = Form(...),
    output_path: Optional[str] = Form('mag/saved.png'),
    parameters: Optional[str] = Form(None)
):
    """
    API endpoint для обработки изображения с параметрами из JSON
    
    Пример запроса:
    {
        "image_path": "path/to/image.jpg",
        "output_path": "path/to/output.png",
        "parameters": {
            "blur_ksize": 5,
            "sobel_ksize": 3,
            "sharp_k": 1.2,
            "morph_ops": {"open": 3, "close": 5}
        }
    }
    """
    try:
        # Устанавливаем параметры если они есть
        if parameters:
            try:
                params_dict = json.loads(parameters)
                processor.set_parameters(**params_dict)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in parameters: {str(e)}"
                )
        
        # Обрабатываем изображение
        pil_image = Image.open(image_path)
        original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        processed = processor.process_image(original_image)
        
        # Сохраняем результат
        processor.save_image(processed, output_path)
        
        return JSONResponse({
            "status": "success",
            "message": f"Image processed and saved to {output_path}",
            "output_path": output_path
        })
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Processing error: {str(e)}"
        )

@app.post('/api/set_parameters')
async def set_parameters_api(parameters: Dict[str, Any]):
    """
    API endpoint для установки параметров обработки
    
    Пример запроса:
    {
        "blur_ksize": 5,
        "sobel_ksize": 3,
        "sharp_k": 1.2,
        "morph_ops": {"open": 3, "close": 5}
    }
    """
    try:
        processor.set_parameters(**parameters)
        return JSONResponse({
            "status": "success",
            "message": "Parameters updated",
            "current_parameters": {
                "blur_ksize": processor.blur_ksize,
                "sobel_ksize": processor.sobel_ksize,
                "sharp_k": processor.sharp_k,
                "alpha": processor.alpha,
                "beta": processor.beta,
                "gamma": processor.gamma,
                "binary_thresh": processor.binary_thresh,
                "morph_ops": processor.morph_ops
            }
        })
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error updating parameters: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)