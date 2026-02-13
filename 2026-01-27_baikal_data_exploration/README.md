# Baikal Data Exploration

**Date:** 2026-01-27
**Goal:** Initial exploration of the dataset `baikal_mc2020_multi_split_0924mid_eq_norm.h5`.
**Data Path:** `/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5`

## Tasks
1. Inspect HDF5 file structure.
2. Load a small sample (first 1000 entries).
3. Calculate basic statistics (mean, std, min, max).
4. Visualize data distribution.

## Outputs
- Console output with file structure and stats.
- Plots saved in `plots/` directory.

Результаты:
   1. Структура файла:
       * Файл содержит группы train, test, val, а также norm_param.
       * Основные данные лежат в подгруппах data/data (фичи) и labels/data (метки).
       * Размер обучающей выборки (train/data/data): 1,431,319,717 записей (5 признаков).
       * Размер валидационной выборки (val/data/data): 24,456,638 записей.
       * Размер тестовой выборки (test/data/data): 85,495,459 записей.

   2. Статистика (по первым 1000 примерам из train):
       * Данные, похоже, нормализованы (среднее близко к 0, стандартное отклонение около 1 для
         большинства признаков).
       * Feature 0: Mean: -0.04, Std: 0.50 (имеет длинный хвост в положительную сторону, max ~8.45).
       * Features 1-4: Mean: ~0.0, Std: ~1.0. Распределения выглядят более симметричными.

   3. Графики:
       * В папке plots/ сохранены гистограммы распределений для каждого из 5 признаков:
           * train_data_feat_0_hist.png ... train_data_feat_4_hist.png

  Все исходные коды находятся в ~/experiments/2026-01-27_baikal_data_exploration/src/.
