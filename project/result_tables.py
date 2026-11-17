"""Таблиці звітів із реально обчислених результатів."""


def f(x, d=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return (f"{x:.2e}" if x != 0 and abs(x) < 10 ** (-d) else f"{x:.{d}f}").replace(
            ".", ","
        )
    return str(x)


def tables(n, r):
    items = []

    def add(title, headers, rows, comment):
        items.append((title, headers, rows, comment))

    add(
        "Розміри незалежних вибірок",
        ["Вибірка", "Спостережень"],
        [[k, str(v)] for k, v in r["split"].items()],
        "Статистики перетворень визначено лише на тренувальній частині. Валідаційні та тестові індекси збережено окремо.",
    )
    if n == 1:
        add(
            "Оцінки трьох оптимізаторів",
            ["Метод", "Train R²", "Val R²", "Test R²", "Test RMSE, USD"],
            [
                [name]
                + [f(v[part]["r2"]) for part in ["train", "validation", "test"]]
                + [f(v["test"]["rmse"], 2)]
                for name, v in r["models"].items()
            ],
            "Метод вибрано за Validation RMSE до оцінювання Test. Значення R² обчислено власною формулою та звірено зі scikit-learn.",
        )
        add(
            "Налаштування оптимізації",
            ["Метод", "α", "Batch", "Епох"],
            [
                [name, f(v["learning_rate"]), v["batch_size"], v["epochs"]]
                for name, v in r["models"].items()
            ],
            "Для кожного методу перевірено три швидкості навчання; збережено кращу за Validation модель. Після 80 епох без поліпшення спрацьовує рання зупинка.",
        )
        add(
            "Діагностика залишків і залежності ознак",
            ["Перевірка", "Значення"],
            [
                ["Shapiro-Wilk p", f(r["shapiro"]["p"], 8)],
                ["Breusch-Pagan p", f(r["breusch_pagan"]["p"], 8)],
                ["Найбільший VIF", f(max(r["vif"].values()), 2)],
                [
                    "Чисельна похибка градієнта",
                    f(r["gradient_check"]["max_relative_error"], 10),
                ],
            ],
            "Малі p-значення вказують на порушення відповідних нульових гіпотез. Це обмежує класичну інтерпретацію коефіцієнтів, але не скасовує фактичну оцінку прогнозу на відкладених даних.",
        )
        add(
            "Додатковий експеримент Ridge",
            ["Показник", "Без штрафу", "Ridge"],
            [
                [
                    "Норма ваг",
                    f(r["ridge"]["norm_unregularized"]),
                    f(r["ridge"]["norm_regularized"]),
                ],
                [
                    "Test RMSE, USD",
                    f(r["models"][r["winner"]]["test"]["rmse"], 2),
                    f(r["ridge"]["test"]["rmse"], 2),
                ],
                [
                    "Test R²",
                    f(r["models"][r["winner"]]["test"]["r2"]),
                    f(r["ridge"]["test"]["r2"]),
                ],
            ],
            f"Коефіцієнт λ={f(r['ridge']['lambda'])} вибрано за Validation. Порівняння не припускає, що штраф обов’язково підвищить точність.",
        )
    elif n == 2:
        add(
            "Глобальні та централізована моделі",
            ["Модель", "Accuracy", "Macro-F1", "Bal. acc.", "Log-loss"],
            [
                [name]
                + [
                    f(v[k])
                    for k in ["accuracy", "macro_f1", "balanced_accuracy", "log_loss"]
                ]
                for name, v in r["global_metrics"].items()
            ],
            "Усі підсумкові оцінки отримано на одному Global Test. Для кожного експерименту кінцеві ваги вибрано за найменшою Global Validation loss.",
        )
        for name, rows in r["local_metrics"].items():
            add(
                "Локальні моделі: " + name,
                ["Клієнт", "n", "Accuracy", "Macro-F1", "Log-loss"],
                [
                    [
                        i + 1,
                        v["n"],
                        f(v["accuracy"]),
                        f(v["macro_f1"]),
                        f(v["log_loss"]),
                    ]
                    for i, v in enumerate(rows)
                ],
                "Локальні моделі навчені тільки на даних відповідного клієнта. На сервер передано їхні параметри, після чого метрики обчислено на Global Test.",
            )
        for name in ["IID", "Dirichlet 5", "Dirichlet 1", "Dirichlet 0.3"]:
            add(
                "Кількість зразків класів у клієнтів: " + name,
                ["Клієнт"] + [str(i) for i in range(7)],
                [[i + 1] + row for i, row in enumerate(r["client_distribution"][name])],
                "Сума всіх клітинок класів дорівнює FL Train Pool. Кожен навчальний індекс належить рівно одному клієнту; перетинів із Global Validation/Test немає.",
            )
        add(
            "Кодування класів",
            ["Код", "Сорт"],
            [[i, name] for i, name in enumerate(r["classes"])],
            "Кодування однакове для всіх клієнтів і сервера. Навіть якщо клас відсутній у клієнта, вихід його локальної моделі має сім компонентів.",
        )
    elif n == 3:
        add(
            "Звичайний стратифікований Test",
            ["Комбінація", "Accuracy", "Macro-F1", "Bal. acc.", "ROC-AUC"],
            [
                [name]
                + [
                    f(v[k])
                    for k in ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc"]
                ]
                for name, v in r["models"].items()
            ],
            "Шість комбінацій не мають помилок на цьому поділі. Це ефект конкретного набору; він не підтверджує дослідницьку гіпотезу про обов’язкову перевагу TF-IDF.",
        )
        add(
            "Окремі теми у Train та Test",
            ["Комбінація", "Accuracy", "Macro-F1", "ROC-AUC"],
            [
                [name] + [f(v[k]) for k in ["accuracy", "macro_f1", "roc_auc"]]
                for name, v in r["group_holdout"]["models"].items()
            ],
            "Груповий поділ має нуль спільних тем. FastText, IDF, скейлер і класифікатори повторно навчено лише на тренувальних темах цього експерименту.",
        )
        add(
            "Аудит даних",
            ["Показник", "Значення"],
            [
                ["Рядків у вихідному CSV", r["raw_rows"]],
                ["Після очищення", r["rows"]],
                ["Прибрано дублікатів", r["duplicates_removed"]],
                ["Тем", r["topics"]],
            ]
            + [[k, v] for k, v in r["generation_methods"].items()],
            "Канонічні дублікати видалено до поділу, суперечливих міток для однакових текстів немає. Метадані виявляють шаблонний спосіб формування корпусу.",
        )
        add(
            "Геометрія початкового простору",
            ["Агрегація", "Silhouette"],
            [[k, f(v)] for k, v in r["silhouette_100d"].items()],
            "Silhouette обчислено на тестових 100-вимірних векторах із відомими класами; він є описовою оцінкою, а не критерієм налаштування моделі.",
        )
    elif n == 4:
        add(
            "Зв’язок пропусків із класом на Train",
            ["Ознака", "Пропуски, %", "Метод", "p"],
            [
                [
                    v["feature"],
                    f(100 * v["missing_fraction"], 2),
                    v["method"],
                    f(v["p"], 6),
                ]
                for v in r["missing_association"]
            ],
            "Для малих очікуваних частот застосовано Fisher із 10000 Monte Carlo вибірками. Виявлена залежність суперечить простому MCAR, проте не розрізняє MAR та MNAR.",
        )
        for flag in ["A", "B"]:
            add(
                "Повторна CV: варіант " + flag,
                ["Модель", "Accuracy ± SD", "Macro-F1 ± SD", "Bal. acc. ± SD"],
                [
                    [name]
                    + [
                        f(r["baseline"][flag + " / " + name][key]["mean"])
                        + " ± "
                        + f(r["baseline"][flag + " / " + name][key]["std"])
                        for key in ["test_acc", "test_f1m", "test_bacc"]
                    ]
                    for name in ["DT", "RF", "XGB"]
                ],
                "Усі оцінки використовують однакові 15 поділів. SD є розкидом по залежних fold, а не довірчим інтервалом для незалежних спостережень.",
            )
        add(
            "Вкладена CV та оптимістична оцінка",
            ["Модель", "Outer macro-F1", "Outer SD", "Best value"],
            [
                [name, f(v["mean"]), f(v["std"]), f(v["ordinary_best_value"])]
                for name, v in r["nested"].items()
            ],
            "Основна оцінка процедури налаштування - зовнішня CV. Best value наведено для порівняння і не використовується як незалежна оцінка якості.",
        )
        for name, params in r["best_params"].items():
            add(
                "Найкращі гіперпараметри: " + name,
                ["Параметр", "Значення"],
                [[k, f(v)] for k, v in params.items()],
                "Параметри знайдено на всій Train після оцінки процедури вкладеною CV; Test у пошуку не використано.",
            )
        add(
            "Скоригований тест Наде-Бенжіо",
            ["Порівняння", "Середня різниця", "t", "p"],
            [
                [name, f(v["mean_difference"]), f(v["t"]), f(v["p"], 6)]
                for name, v in r["statistical_tests"].items()
            ],
            "Рівень окремої перевірки α=0,05. Невідхилення нульової гіпотези не означає доведену рівність моделей.",
        )
        add(
            "Фінальний Test",
            ["Модель", "Accuracy", "Macro-F1", "Bal. acc."],
            [
                [name]
                + [f(v[k]) for k in ["accuracy", "macro_f1", "balanced_accuracy"]]
                for name, v in r["test"].items()
            ],
            "Найкращу модель обрано за вкладеною CV, не за цією таблицею. Для кожної моделі Test оцінено після фіксації гіперпараметрів.",
        )
        add(
            "Classification report вибраної моделі",
            ["Клас", "Precision", "Recall", "F1", "Support"],
            [
                [c]
                + [
                    f(r["classification_report"][c][k])
                    for k in ["precision", "recall", "f1-score"]
                ]
                + [int(r["classification_report"][c]["support"])]
                for c in r["classes"]
            ],
            "Ці оцінки показують якість кожного класу окремо. Для рідкісного класу мала тестова підтримка робить метрику особливо нестійкою.",
        )
        add(
            "Об’єкти розбіжності",
            ["Рядок", "Клас", "DT", "RF", "XGB"],
            [
                [row[k] for k in ["row", "true", "DT", "RF", "XGB"]]
                for row in r["disagreements"][:15]
            ],
            f"У Test знайдено {r['disagreement_count']} об’єктів розбіжності. Повний список міститься у disagreements.csv; тут наведено не більш як 15.",
        )
        exp = r["local_explanation"]
        add(
            "Початкові значення пояснюваного об’єкта",
            ["Ознака", "Значення"],
            [
                [k, "ПРОПУСК" if v is None else f(v)]
                for k, v in exp["raw_features"].items()
            ],
            f"Об’єкт вибрано з {exp['source']}; індекс рядка з нуля {exp['original_row']}, справжній клас {exp['true_class']}. Це значення до імпутації.",
        )
        imputed = sorted(
            set().union(*(v["imputed_values"] for v in exp["models"].values()))
        )
        add(
            "Заповнення пропусків пояснюваного об’єкта",
            ["Ознака", "DT", "RF", "XGB"],
            [
                [c]
                + [
                    f(exp["models"][name]["imputed_values"].get(c, "Вилучено"))
                    for name in ["DT", "RF", "XGB"]
                ]
                for c in imputed
            ],
            "Значення імпутації обчислені на Train відповідної моделі. «Вилучено» означає константну або повністю порожню ознаку, яку прибрав навчальний трансформер.",
        )
        add(
            "Ймовірності класів пояснюваного об’єкта",
            ["Клас", "DT", "RF", "XGB"],
            [
                [c]
                + [
                    f(exp["models"][name]["probabilities"][i])
                    for name in ["DT", "RF", "XGB"]
                ]
                for i, c in enumerate(r["classes"])
            ],
            "Порівняння ймовірностей допомагає відрізнити невпевнені прогнози поблизу межі класів від жорсткого рішення окремого дерева.",
        )
        for name, ex in exp["models"].items():
            add(
                "П’ять головних SHAP-внесків: " + name,
                ["Перетворена ознака", "Значення", "SHAP"],
                [
                    [row["feature"], f(row["transformed_value"]), f(row["shap"])]
                    for row in ex["top_contributions"]
                ],
                f"Прогнозований клас {ex['predicted_class']}; масштаб {ex['output_scale']}. Похибка відновлення суми внесків {ex['additivity_error']:.3g}.",
            )
    elif n == 5:
        add(
            "Порівняння всіх конфігурацій",
            ["Метод", "Dead, %", "Sparsity, %", "Val acc.", "Val CE"],
            [
                [
                    name,
                    f(v["dead_percent"], 2),
                    f(v["sparsity_percent"], 2),
                    f(v["validation_accuracy"]),
                    f(v["validation_loss"]),
                ]
                for name, v in r["experiments"].items()
            ],
            "Порівнюється фінальна якість після 30 епох на однакових даних та з однаковими початковими вагами. Dead обчислено без Dropout на всій Train; sparsity використовує поріг |w|<0,01.",
        )
        add(
            "Мертві нейрони за прихованими шарами",
            ["Метод", "Шар 1 / 64", "Шар 2 / 48", "Шар 3 / 32"],
            [[name] + v["dead_per_layer"] for name, v in r["experiments"].items()],
            "Для кожного нейрона перевірено максимум передактивації по всіх 12000 навчальних зображеннях. Лише максимум ≤0 означає мертвий нейрон на цій вибірці.",
        )
        add(
            "Перевірки та фінальний Test",
            ["Показник", "Значення"],
            [
                [
                    "Перевірено параметрів градієнта",
                    r["gradient_check"]["parameters_checked"],
                ],
                [
                    "Найбільша відносна похибка",
                    f(r["gradient_check"]["max_relative_error"], 12),
                ],
                ["Похибка відновлення ймовірностей", r["restore_max_error"]],
                ["Вибрана конфігурація", r["winner"]],
                ["Test accuracy", f(r["test"]["accuracy"])],
                ["Test macro-F1", f(r["test"]["macro_f1"])],
            ],
            "Конфігурацію вибрано за Validation CE. Серіалізація зберігає конфігурацію та всі ваги; вихід відновленої моделі має збігатися з оригіналом до 10⁻⁷.",
        )
        add(
            "Ефект clipping",
            ["Метод", "Ваги на ±1", "Обрізані кроки", "Усього кроків"],
            [
                [
                    name,
                    v["weights_at_boundary"],
                    v["gradient_clipped_steps"],
                    v["total_steps"],
                ]
                for name, v in r["experiments"].items()
                if "clip" in name
            ],
            "Weight clipping і gradient clipping діють на різні величини. Кількість фактичних обрізань дозволяє перевірити, чи було обмеження активним.",
        )
    return items


def conclusions(n, r):
    if n == 1:
        v = r["models"][r["winner"]]["test"]
        return [
            f"За Validation RMSE вибрав {r['winner']}. На Test отримав R²={f(v['r2'])}, RMSE={f(v['rmse'], 2)} USD і MAE={f(v['mae'], 2)} USD. Результат одержано власним градієнтним навчанням, без Normal Equation.",
            f"Найбільший VIF дорівнює {f(max(r['vif'].values()), 2)}. Перевірки залишків і графіки показують обмеження класичної лінійної інтерпретації, тому якість прогнозу й достовірність окремих коефіцієнтів не ототожнюю. Власні градієнти пройшли чисельну перевірку.",
        ]
    if n == 2:
        a = r["global_metrics"]
        return [
            f"Централізована модель отримала Test macro-F1={f(a['central']['macro_f1'])}. Для IID FedAvg значення становить {f(a['IID']['macro_f1'])}, а для Dirichlet α=0,3 і E=1 - {f(a['Dirichlet 0.3']['macro_f1'])}.",
            f"За α=0,3 зміна E з 1 до 10 дала macro-F1={f(a['Dirichlet 0.3, E=10']['macro_f1'])}. Висновок обмежений однією відтворюваною реалізацією розподілу й однаковими 60 раундами, а не рівними витратами обчислень. FedAvg, softmax та градієнти реалізовано NumPy; сирі дані клієнтів у серверному класі не зберігаються.",
        ]
    if n == 3:
        return [
            f"Після очищення використав {r['rows']} текстів. На стратифікованому Test усі шість комбінацій отримали macro-F1=1,0000. Тому цей експеримент не показує переваги TF-IDF за F1 на звичайному поділі.",
            f"Додатковий груповий поділ не має спільних тем: {r['group_holdout']['train_topics']} у Train і {r['group_holdout']['test_topics']} у Test. Виявлено шаблонні способи створення даних, тому високі оцінки не переношу на довільні тексти й не трактую як підтвердження авторства конкретної людини.",
        ]
    if n == 4:
        name = r["winner_nested_cv"]
        v = r["nested"][name]
        changes = {
            m: r["baseline"]["B / " + m]["test_f1m"]["mean"]
            - r["baseline"]["A / " + m]["test_f1m"]["mean"]
            for m in ["DT", "RF", "XGB"]
        }
        increases = {
            m: r["nested"][m]["mean"] - r["baseline"]["B / " + m]["test_f1m"]["mean"]
            for m in ["DT", "RF", "XGB"]
        }
        improved = max(increases, key=increases.get)
        hardest = min(
            r["classes"], key=lambda c: r["classification_report"][c]["recall"]
        )
        tests = "; ".join(
            k + ": p=" + f(value["p"], 6) for k, value in r["statistical_tests"].items()
        )
        return [
            f"За зовнішньою CV вибрав {name}: macro-F1={f(v['mean'])} ± {f(v['std'])}. Фінальний Test macro-F1={f(r['test'][name]['macro_f1'])}; він не використовувався для вибору моделі. Індикатори пропусків змінили середній baseline macro-F1 на "
            + ", ".join(m + ": " + f(changes[m]) for m in changes)
            + ".",
            f"Скориговані парні тести дали {tests}. Результат кожної перевірки порівнюю з α=0,05; ці p-значення не доводять універсальної переваги моделі на інших даних. Для вибраної моделі best_value={f(v['ordinary_best_value'])}, тоді як незалежна зовнішня оцінка={f(v['mean'])}.",
            f"Найбільша описова різниця між tuned outer CV та baseline B CV спостерігалася для {improved}: {f(increases[improved])}; це різні схеми поділів, тому така різниця не є парним тестом приросту. Найменший Test recall має клас {hardest}: {f(r['classification_report'][hardest]['recall'])} за support={int(r['classification_report'][hardest]['support'])}. У Test є {r['disagreement_count']} об’єктів розбіжності; локальні SHAP-внески перевірено за властивістю адитивності.",
            f"Для практичного використання обрав би {name} як переможця за заздалегідь обраною процедурою nested CV, врахувавши її розкид і витрати навчання. Залежність пропусків від класу та важливості індикаторів аналізую разом; самі ці дані не дають однозначно розділити MAR та MNAR.",
        ]
    if n == 5:
        best = r["experiments"][r["winner"]]
        return [
            f"Виконав 13 контрольованих експериментів із мережею 784-64-48-32-10. Найменшу фінальну Validation CE отримала конфігурація {r['winner']}: validation accuracy={f(best['validation_accuracy'])}. Її Test accuracy={f(r['test']['accuracy'])}.",
            f"Максимальна відносна похибка чисельної перевірки градієнтів {r['gradient_check']['max_relative_error']:.3g}; відновлення ваг дало максимальну різницю ймовірностей {r['restore_max_error']}. Сильніша регуляризація в цій постановці часто погіршувала результат; частка малих ваг і кількість мертвих нейронів не є взаємозамінними показниками.",
        ]
