# 🚀 Inteligentni Sistem za Praćenje i Analizu Ponašanja v2.0

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5c3ee8?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet?logo=ultralytics)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Pretvorite pasivni video nadzor u proaktivni, inteligentni sistem za prikupljanje i analizu podataka u realnom vremenu.**

Ovo je sveobuhvatna platforma za video analitiku koja koristi najmodernije algoritme mašinskog učenja kako bi transformisala sirove video snimke u dragocene poslovne i bezbednosne uvide. Kroz fuziju **YOLOv8**, state-of-the-art modela za detekciju objekata, i **DeepSORT** algoritma za robusno praćenje, ovaj sistem nudi neuporedivu preciznost, brzinu i dubinu analize kretanja i interakcija u bilo kom okruženju.

<br>

![GIF Demonstracija Sistema](https://user-images.githubusercontent.com/26892694/167894676-3c2f0f44-8839-4c8c-95b6-7243c5332b7e.gif)
*(GIF koji prikazuje praćenje više objekata, dodele ID-jeva i presecanje definisanih linija)*

---

## 📋 Sadržaj

* [Tehnološki Skup](#-tehnološki-skup)
* [Ključne Mogućnosti](#-ključne-mogućnosti)
* [Instalacija i Podešavanje](#-instalacija-i-podešavanje)
* [Konfiguracija Sistema](#️-konfiguracija-sistema)
* [Pokretanje Sistema](#-pokretanje-sistema)
* [Potencijalne Primene](#-potencijalne-primene)
* [Plan Razvoja (Roadmap)](#-plan-razvoja-roadmap)
* [Doprinos](#-doprinos)
* [Licenca](#-licenca)

---

## 🛠️ Tehnološki Skup

Sistem je izgrađen na temeljima proverenih i moćnih tehnologija iz sveta Python-a i veštačke inteligencije:

* **Python 3.9+:** Glavni programski jezik, izabran zbog svoje fleksibilnosti, ogromnog ekosistema i podrške za AI/ML biblioteke.
* **Ultralytics YOLOv8:** Najnovija generacija YOLO (You Only Look Once) modela, pruža izuzetno brzu i tačnu detekciju objekata u realnom vremenu.
* **DeepSORT:** Algoritam za praćenje koji rešava problem okluzije (preklapanja objekata) i ponovne identifikacije, dodeljujući jedinstveni ID svakom praćenom objektu.
* **OpenCV:** Defakto standard za računarsku viziju; koristi se za čitanje video stream-ova, manipulaciju slikama i iscrtavanje rezultata.
* **PyTorch:** Osnovni deep learning framework na kojem je izgrađen YOLOv8, omogućava hardversku akceleraciju (GPU) za maksimalne performanse.
* **NumPy:** Fundamentalna biblioteka za numeričke operacije, neophodna za efikasnu manipulaciju podacima o pozicijama i putanjama.

---

## ✨ Ključne Mogućnosti

### 🎯 Detekcija i Praćenje
* **📈 Visoko-precizno Praćenje:** Robusno praćenje više objekata istovremeno, održavajući jedinstvene ID-jeve čak i u uslovima gustog saobraćaja, preklapanja i kratkotrajnih izlazaka iz kadra.
* **📦 Višeklasna Detekcija:** Sistem nije ograničen na ljude. Zahvaljujući snazi YOLOv8 modela pre-treniranog na COCO setu podataka, sposoban je da detektuje i prati objekte iz **80 unapred definisanih klasa**, uključujući:
## 👥 Ljudi i Povezani Predmeti
* `person`
* `backpack`
* `handbag`
* `suitcase`
* `umbrella`

---

## 🚗 Saobraćaj i Vozila
* `bicycle`
* `car`
* `motorcycle`
* `bus`
* `train`
* `truck`
* `boat`

---

## 🚦 Ulični Signali i Objekti
* `traffic light`
* `fire hydrant`
* `stop sign`
* `parking meter`
* `bench`

---

## 🐾 Životinje (domaće i divlje)
* `bird`
* `cat`
* `dog`
* `horse`
* `sheep`
* `cow`
* `elephant`
* `bear`
* `zebra`
* `giraffe`

---

## 🪑 Nameštaj i Kućni Aparati
* `chair`
* `couch`
* `potted plant`
* `bed`
* `dining table`
* `toilet`
* `tv`
* `laptop`
* `mouse`
* `remote`
* `keyboard`
* `cell phone`
* `microwave`
* `oven`
* `refrigerator`

---

## 🍽️ Hrana i Kuhinjski Pribor
* `bottle`
* `wine glass`
* `cup`
* `fork`
* `knife`
* `spoon`
* `bowl`
* `banana`
* `apple`
* `sandwich`
* `pizza`
* `donut`
* `cake`

---

## 🏀 Sportska Oprema
* `frisbee`
* `skis`
* `snowboard`
* `sports ball`
* `kite`
* `baseball bat`
* `skateboard`
* `tennis racket`

### 🧠 Napredna Analiza Ponašanja
* **🔥 Heatmape (Mape Toplote):** Generisanje dinamičkih ili kumulativnih heatmapi koje vizualizuju "vruće" zone – područja najviše aktivnosti i zadržavanja. Idealno za analizu kretanja kupaca ili identifikaciju uskih grla.
* ** เส้นทาง Putanje i Tragovi:** Vizualizacija putanja svakog praćenog objekta, omogućavajući detaljnu analizu tokova kretanja.
* **⚠️ Detekcija Anomalija:** Modularni sistem za prepoznavanje sumnjivih ili specifičnih obrazaca ponašanja:
    * **Zadržavanje (Loitering):** Alarmiranje kada se objekat zadrži u definisanoj zoni duže od dozvoljenog vremena.
    * **Prekoračenje Brzine:** Detekcija objekata koji se kreću brže od definisanog praga.
    * **Kretanje u Zabranjenom Smeru:** Prepoznavanje kretanja suprotnog od očekivanog.
* **👨‍👩‍👧‍👦 Detekcija Grupa:** Identifikacija formiranja, kretanja i rasipanja grupa u realnom vremenu.

### ⚙️ Fleksibilnost i Upravljanje
* **YAML Konfiguracija:** Potpuna prilagodljivost kroz intuitivne YAML fajlove. Definišite video izvore, zone interesa (linije, poligone), pragove za alarme i module koje želite da aktivirate bez izmene koda.
* **📹 Univerzalna Kompatibilnost Izvora:** Obradite snimke iz lokalnih video fajlova (`.mp4`, `.avi`, `.mov`), IP kamera i live **RTSP** stream-ova.
* **📝 Detaljno Logovanje i Izveštavanje:** Svaki bitan događaj (ulaz/izlaz iz zone, detektovana anomalija, prebrojavanje) se beleži u strukturiranom formatu (CSV, JSON) sa vremenskom oznakom i ID-jem objekta. Na kraju obrade, generiše se sumarni statistički izveštaj.
* **🚀 Merenje Performansi:** Pratite ključne metrike sistema (FPS, vreme obrade po frejmu) da biste optimizovali rad na različitim hardverima.
