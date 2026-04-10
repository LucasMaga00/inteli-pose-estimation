# inteli-pose-estimation

Estimativa de pose para bovinos com base no **ANIMAL-POSE DATASET** e em um pipeline inspirado em tutoriais de pose estimation.

---

## Notas do dataset (para não esquecer)

- O arquivo **`keypoints.json`** corresponde ao conjunto **Part I** do Animal-Pose (anotações com keypoints para cinco categorias, incluindo **vaca / `cow`**, `category_id` 5). No JSON há **4.608** imagens referenciadas e **6.117** anotações; entre elas, **842** anotações de bovinos.
- A pasta de imagens está nomeada como **`animalpose_image_part2`**, mas o conteúdo é o layout típico da **Part I** (subpastas `dog/`, `cat/`, `cow/`, `horse/`, `sheep/`), **não** a Part II (apenas bounding boxes). O nome da pasta é enganoso; o que importa é o conteúdo.
- **Disponibilidade parcial:** no repositório local existem apenas **200 imagens por espécie** (1.000 imagens no total), enquanto o JSON descreve o pacote completo. Para **bovinos**, apenas cerca de **200** imagens em `cow/` possuem arquivo correspondente às entradas do JSON; as demais anotações de vaca no JSON referem-se a arquivos que ainda **não** estão na pasta.
- **Escopo atual do trabalho:** vamos trabalhar **somente com o subconjunto de bovinos cujo arquivo existe em disco** (junção entre `keypoints.json` e os nomes de arquivo presentes em `animalpose_image_part2/cow/`). O pipeline deve fazer essa **interseção por arquivo** para não quebrar quando mais imagens forem baixadas depois.
- **Limitação a documentar** na versão final do README: resultados e estatísticas valem para esse **subconjunto**, não para o dataset Animal-Pose completo; generalização pode ser mais fraca até que o download completo das imagens alinhadas ao JSON seja obtido.

---

*Seções adicionais (EDA, filtragem, figuras de processamento, resultados e conclusões) serão preenchidas conforme o andamento do projeto.*
