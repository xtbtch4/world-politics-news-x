# World Politics News → X

Автоматический сбор важных политических и мировых новостей с публикацией в X.

## Как работает

- каждый час проверяет международные RSS-источники;
- принимает только материалы, опубликованные за последние 7 дней;
- оценивает важность события по теме, свежести и надёжности источника;
- переводит заголовок на русский язык;
- публикует не более двух новых новостей за один запуск;
- добавляет источник и прямую ссылку;
- сохраняет историю ссылок и заголовков, чтобы не создавать повторы.

Источники: BBC, The Guardian, Al Jazeera, DW, France 24, NPR, UN News, POLITICO Europe и The Kyiv Independent.

## Обязательные GitHub Secrets

Откройте `Settings → Secrets and variables → Actions → New repository secret` и по очереди создайте:

| Название | Значение из X Developer Console |
|---|---|
| `X_CONSUMER_KEY` | Consumer Key |
| `X_CONSUMER_SECRET` | Consumer Key Secret |
| `X_ACCESS_TOKEN` | Access Token с правами Read and write |
| `X_ACCESS_TOKEN_SECRET` | Access Token Secret |

Не добавляйте ключи непосредственно в файлы репозитория.

## Первый запуск

1. Убедитесь, что в X Developer Console подключены API-кредиты для Pay Per Use.
2. Откройте вкладку `Actions`.
3. Выберите `Publish world news to X`.
4. Нажмите `Run workflow`.
5. Откройте выполненный запуск и проверьте шаг `Collect and publish news`.

Автоматическое расписание: каждый час, на 17-й минуте часа. GitHub иногда запускает плановые задания с небольшой задержкой.

## Настройки

Параметры находятся в `.github/workflows/publish.yml`:

- `MAX_AGE_DAYS` — максимальный возраст новости;
- `MAX_POSTS_PER_RUN` — максимум публикаций за запуск;
- `MIN_IMPORTANCE_SCORE` — порог важности.

История публикаций хранится в `data/posted.json` и автоматически обновляется после успешной публикации.
