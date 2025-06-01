use actix_web::{App, HttpResponse, HttpServer, Responder, Result, web};
use clap::{Parser, Subcommand};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use tch::{Device, Kind, Tensor, nn, nn::Module};

// Структура для запроса предсказания
#[derive(Deserialize)]
struct PredictRequest {
    image: Vec<f32>, // Входное изображение как плоский вектор
}

// Структура для ответа с предсказанием
#[derive(Serialize)]
struct PredictResponse {
    class: i64,
    confidence: f32,
}

// Структура для ответа с предсказанием ошибок ОС
#[derive(Serialize)]
struct OsErrorPredictResponse {
    error_type: String,
    os_type: String,
    confidence: f32,
    description: String,
}

// Типы ошибок операционных систем
const OS_ERROR_TYPES: &[&str] = &[
    "blue_screen_of_death", // Синий экран смерти Windows
    "kernel_panic",         // Паника ядра Linux/macOS
    "application_crash",    // Краш приложения
    "memory_error",         // Ошибка памяти
    "disk_error",           // Ошибка диска
    "network_error",        // Сетевая ошибка
    "permission_denied",    // Отказ в доступе
    "file_not_found",       // Файл не найден
    "system_overload",      // Перегрузка системы
    "driver_error",         // Ошибка драйвера
];

const OS_TYPES: &[&str] = &["windows", "linux", "macos", "unknown"];

// Упрощенная модель CNN для демонстрации
fn simple_cnn(p: &nn::Path, num_classes: i64) -> impl nn::Module {
    let conv1 = nn::conv2d(
        p / "conv1",
        3,
        32,
        3,
        nn::ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let conv2 = nn::conv2d(
        p / "conv2",
        32,
        64,
        3,
        nn::ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let fc1 = nn::linear(p / "fc1", 64 * 8 * 8, 128, Default::default());
    let fc2 = nn::linear(p / "fc2", 128, num_classes, Default::default());

    nn::seq()
        .add(conv1)
        .add_fn(|xs| {
            xs.relu()
                .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
        })
        .add(conv2)
        .add_fn(|xs| {
            xs.relu()
                .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
        })
        .add_fn(|xs| xs.flatten(1, -1))
        .add(fc1)
        .add_fn(|xs| xs.relu())
        .add(fc2)
}

// Специализированная модель CNN для анализа ошибок ОС с более крупными изображениями
fn os_error_cnn(
    p: &nn::Path,
    num_error_types: i64,
    num_os_types: i64,
) -> (impl nn::Module, impl nn::Module) {
    // Общие сверточные слои для извлечения признаков
    let conv1 = nn::conv2d(
        p / "conv1",
        3,
        64,
        5,
        nn::ConvConfig {
            padding: 2,
            ..Default::default()
        },
    );
    let conv2 = nn::conv2d(
        p / "conv2",
        64,
        128,
        5,
        nn::ConvConfig {
            padding: 2,
            ..Default::default()
        },
    );
    let conv3 = nn::conv2d(
        p / "conv3",
        128,
        256,
        3,
        nn::ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );

    let shared_features = nn::seq()
        .add(conv1)
        .add_fn(|xs| {
            xs.relu()
                .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
        })
        .add(conv2)
        .add_fn(|xs| {
            xs.relu()
                .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
        })
        .add(conv3)
        .add_fn(|xs| {
            xs.relu()
                .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
        })
        .add_fn(|xs| xs.flatten(1, -1));

    // Классификатор типа ошибки
    let error_classifier = nn::seq()
        .add(nn::linear(
            p / "error_fc1",
            256 * 16 * 16,
            512,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::dropout(p / "error_dropout", 0.5))
        .add(nn::linear(p / "error_fc2", 512, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            p / "error_out",
            256,
            num_error_types,
            Default::default(),
        ));

    // Классификатор типа ОС
    let os_classifier = nn::seq()
        .add(nn::linear(
            p / "os_fc1",
            256 * 16 * 16,
            256,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::dropout(p / "os_dropout", 0.3))
        .add(nn::linear(
            p / "os_out",
            256,
            num_os_types,
            Default::default(),
        ));

    (
        nn::seq().add(shared_features).add(error_classifier),
        nn::seq().add_fn(move |xs| {
            let features = xs.apply(&shared_features);
            features.apply(&os_classifier)
        }),
    )
}

// Создание тестовых данных для демонстрации
fn create_dummy_data() -> (Tensor, Tensor) {
    let device = Device::Cpu;
    let train_images = Tensor::randn(&[100, 3, 32, 32], (Kind::Float, device));
    let train_labels = Tensor::randint(10, &[100], (Kind::Int64, device));
    (train_images, train_labels)
}

// Обучение модели
fn train_model(
    model: &dyn nn::Module,
    train_images: &Tensor,
    train_labels: &Tensor,
    vs: &nn::VarStore,
) {
    let mut optimizer = nn::Adam::default().build(vs, 1e-3).unwrap();

    for epoch in 1..=5 {
        let output = model.forward(train_images);
        let loss = output.cross_entropy_for_logits(train_labels);
        optimizer.backward_step(&loss);
        println!("Epoch: {}, Loss: {:.4}", epoch, f64::from(&loss));
    }

    vs.save("model.pt").unwrap();
    println!("Модель сохранена в model.pt");
}

// Веб-обработчик для предсказания
async fn predict(
    req: web::Json<PredictRequest>,
    model_data: web::Data<Mutex<(Box<dyn nn::Module + Send>, nn::VarStore)>>,
) -> Result<impl Responder> {
    let image = Tensor::of_slice(&req.image)
        .to_device(Device::Cpu)
        .view([1, 3, 32, 32]);

    let (model, _) = &*model_data.lock().unwrap();
    let output = model.forward(&image);
    let probs = output.softmax(-1, Kind::Float);
    let (confidence, class) = probs.max_dim(-1, false);

    Ok(HttpResponse::Ok().json(PredictResponse {
        class: i64::from(&class.get(0)),
        confidence: f32::from(&confidence.get(0)),
    }))
}

// Веб-обработчик для предсказания ошибок ОС
async fn predict_os_error(
    req: web::Json<PredictRequest>,
    model_data: web::Data<
        Mutex<(
            (Box<dyn nn::Module + Send>, Box<dyn nn::Module + Send>),
            nn::VarStore,
        )>,
    >,
) -> Result<impl Responder> {
    let image = Tensor::of_slice(&req.image)
        .to_device(Device::Cpu)
        .view([1, 3, 128, 128]);

    let ((error_model, os_model), _) = &*model_data.lock().unwrap();

    // Предсказание типа ошибки
    let error_output = error_model.forward(&image);
    let error_probs = error_output.softmax(-1, Kind::Float);
    let (error_confidence, error_class) = error_probs.max_dim(-1, false);

    // Предсказание типа ОС
    let os_output = os_model.forward(&image);
    let os_probs = os_output.softmax(-1, Kind::Float);
    let (_, os_class) = os_probs.max_dim(-1, false);

    let error_idx = i64::from(&error_class.get(0)) as usize;
    let os_idx = i64::from(&os_class.get(0)) as usize;

    let error_type = OS_ERROR_TYPES
        .get(error_idx)
        .unwrap_or(&"unknown")
        .to_string();
    let os_type = OS_TYPES.get(os_idx).unwrap_or(&"unknown").to_string();

    let description = match error_type.as_str() {
        "blue_screen_of_death" => "Критическая системная ошибка Windows (BSOD)",
        "kernel_panic" => "Критическая ошибка ядра Linux/macOS",
        "application_crash" => "Неожиданное завершение работы приложения",
        "memory_error" => "Ошибка доступа к памяти или нехватка RAM",
        "disk_error" => "Ошибка чтения/записи диска",
        "network_error" => "Проблемы с сетевым подключением",
        "permission_denied" => "Недостаточно прав для выполнения операции",
        "file_not_found" => "Файл или ресурс не найден",
        "system_overload" => "Перегрузка системы",
        "driver_error" => "Ошибка драйвера устройства",
        _ => "Неизвестная ошибка",
    }
    .to_string();

    Ok(HttpResponse::Ok().json(OsErrorPredictResponse {
        error_type,
        os_type,
        confidence: f32::from(&error_confidence.get(0)),
        description,
    }))
}

// Определение аргументов командной строки
#[derive(Parser)]
#[clap(name = "image-classifier")]
#[clap(about = "Утилита для классификации изображений")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Запустить веб-сервер
    Server,
    /// Обучить модель
    Train,
    /// Предсказать класс изображения
    Predict {
        /// Путь к обученной модели
        #[clap(short, long, default_value = "model.pt")]
        model: String,
        /// Путь к изображению для предсказания
        #[clap(short, long)]
        image: String,
    },
    /// Предсказать ошибку операционной системы по скриншоту
    PredictOsError {
        /// Путь к модели для предсказания ошибок ОС
        #[clap(short, long, default_value = "os_error_model.pt")]
        model: String,
        /// Путь к скриншоту с ошибкой
        #[clap(short, long)]
        screenshot: String,
    },
    /// Обучить модель для предсказания ошибок ОС
    TrainOsError,
}

// Функция для загрузки и предобработки изображения
fn load_image<P: AsRef<Path>>(path: P) -> Result<Tensor, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let img = img.resize_exact(32, 32, image::imageops::FilterType::Lanczos3);
    let mut flat = Vec::with_capacity(3 * 32 * 32);

    for y in 0..32 {
        for x in 0..32 {
            let pixel = img.get_pixel(x, y);
            flat.push(pixel[0] as f32 / 255.0);
            flat.push(pixel[1] as f32 / 255.0);
            flat.push(pixel[2] as f32 / 255.0);
        }
    }

    Ok(Tensor::of_slice(&flat)
        .view([1, 3, 32, 32])
        .to_device(Device::Cpu))
}

// Функция для предсказания с использованием утилиты командной строки
fn predict_from_cli(model_path: &str, image_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let model = simple_cnn(&vs.root(), 10);

    vs.load(model_path)?;

    let image_tensor = load_image(image_path)?;
    let output = model.forward(&image_tensor);
    let probs = output.softmax(-1, Kind::Float);
    let (confidence, class) = probs.max_dim(-1, false);

    println!("Предсказанный класс: {}", i64::from(&class.get(0)));
    println!("Уверенность: {:.2}%", f32::from(&confidence.get(0)) * 100.0);

    Ok(())
}

// Функция для загрузки изображения большего размера для анализа ошибок ОС
fn load_screenshot<P: AsRef<Path>>(path: P) -> Result<Tensor, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let img = img.resize_exact(128, 128, image::imageops::FilterType::Lanczos3);
    let mut flat = Vec::with_capacity(3 * 128 * 128);

    for y in 0..128 {
        for x in 0..128 {
            let pixel = img.get_pixel(x, y);
            flat.push(pixel[0] as f32 / 255.0);
            flat.push(pixel[1] as f32 / 255.0);
            flat.push(pixel[2] as f32 / 255.0);
        }
    }

    Ok(Tensor::of_slice(&flat)
        .view([1, 3, 128, 128])
        .to_device(Device::Cpu))
}

// Создание тестовых данных для ошибок ОС
fn create_os_error_dummy_data() -> (Tensor, Tensor, Tensor) {
    let device = Device::Cpu;
    let train_images = Tensor::randn(&[200, 3, 128, 128], (Kind::Float, device));
    let error_labels = Tensor::randint(OS_ERROR_TYPES.len() as i64, &[200], (Kind::Int64, device));
    let os_labels = Tensor::randint(OS_TYPES.len() as i64, &[200], (Kind::Int64, device));
    (train_images, error_labels, os_labels)
}

// Обучение модели для ошибок ОС
fn train_os_error_model() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let (error_model, os_model) = os_error_cnn(
        &vs.root(),
        OS_ERROR_TYPES.len() as i64,
        OS_TYPES.len() as i64,
    );

    let mut optimizer = nn::Adam::default().build(&vs, 1e-4)?;
    let (train_images, error_labels, os_labels) = create_os_error_dummy_data();

    for epoch in 1..=10 {
        let error_output = error_model.forward(&train_images);
        let os_output = os_model.forward(&train_images);

        let error_loss = error_output.cross_entropy_for_logits(&error_labels);
        let os_loss = os_output.cross_entropy_for_logits(&os_labels);
        let total_loss = error_loss + os_loss * 0.5; // Взвешенная потеря

        optimizer.backward_step(&total_loss);

        if epoch % 2 == 0 {
            println!(
                "Epoch: {}, Error Loss: {:.4}, OS Loss: {:.4}, Total Loss: {:.4}",
                epoch,
                f64::from(&error_loss),
                f64::from(&os_loss),
                f64::from(&total_loss)
            );
        }
    }

    vs.save("os_error_model.pt")?;
    println!("Модель для предсказания ошибок ОС сохранена в os_error_model.pt");
    Ok(())
}

// Функция для предсказания ошибок ОС
fn predict_os_error_from_cli(
    model_path: &str,
    screenshot_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let (error_model, os_model) = os_error_cnn(
        &vs.root(),
        OS_ERROR_TYPES.len() as i64,
        OS_TYPES.len() as i64,
    );

    vs.load(model_path)?;

    let image_tensor = load_screenshot(screenshot_path)?;

    // Предсказание типа ошибки
    let error_output = error_model.forward(&image_tensor);
    let error_probs = error_output.softmax(-1, Kind::Float);
    let (error_confidence, error_class) = error_probs.max_dim(-1, false);

    // Предсказание типа ОС
    let os_output = os_model.forward(&image_tensor);
    let os_probs = os_output.softmax(-1, Kind::Float);
    let (os_confidence, os_class) = os_probs.max_dim(-1, false);

    let error_idx = i64::from(&error_class.get(0)) as usize;
    let os_idx = i64::from(&os_class.get(0)) as usize;

    let error_type = OS_ERROR_TYPES.get(error_idx).unwrap_or(&"unknown");
    let os_type = OS_TYPES.get(os_idx).unwrap_or(&"unknown");

    // Описания ошибок
    let description = match *error_type {
        "blue_screen_of_death" => "Критическая системная ошибка Windows (BSOD)",
        "kernel_panic" => "Критическая ошибка ядра Linux/macOS",
        "application_crash" => "Неожиданное завершение работы приложения",
        "memory_error" => "Ошибка доступа к памяти или нехватка RAM",
        "disk_error" => "Ошибка чтения/записи диска",
        "network_error" => "Проблемы с сетевым подключением",
        "permission_denied" => "Недостаточно прав для выполнения операции",
        "file_not_found" => "Файл или ресурс не найден",
        "system_overload" => "Перегрузка системы",
        "driver_error" => "Ошибка драйвера устройства",
        _ => "Неизвестная ошибка",
    };

    println!("=== Анализ ошибки операционной системы ===");
    println!("Тип ошибки: {}", error_type);
    println!("Операционная система: {}", os_type);
    println!(
        "Уверенность (ошибка): {:.2}%",
        f32::from(&error_confidence.get(0)) * 100.0
    );
    println!(
        "Уверенность (ОС): {:.2}%",
        f32::from(&os_confidence.get(0)) * 100.0
    );
    println!("Описание: {}", description);

    Ok(())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Server => {
            println!("Запуск веб-сервера на http://0.0.0.0:5000");

            let device = Device::Cpu;
            let vs = nn::VarStore::new(device);
            let model = simple_cnn(&vs.root(), 10);

            // Попытка загрузить существующую модель или создать новую
            if vs.load("model.pt").is_err() {
                println!("Модель не найдена, создание новой...");
                let (train_images, train_labels) = create_dummy_data();
                train_model(&model, &train_images, &train_labels, &vs);
            }

            let model_data = web::Data::new(Mutex::new((
                Box::new(model) as Box<dyn nn::Module + Send>,
                vs,
            )));

            // Инициализация модели для ошибок ОС
            let device = Device::Cpu;
            let vs_os = nn::VarStore::new(device);
            let (error_model, os_model) = os_error_cnn(
                &vs_os.root(),
                OS_ERROR_TYPES.len() as i64,
                OS_TYPES.len() as i64,
            );

            // Попытка загрузить модель для ошибок ОС
            if vs_os.load("os_error_model.pt").is_err() {
                println!("Модель для ошибок ОС не найдена, создание новой...");
                match train_os_error_model() {
                    Ok(_) => println!("Модель для ошибок ОС создана успешно"),
                    Err(e) => println!("Ошибка создания модели для ошибок ОС: {}", e),
                }
                let _ = vs_os.load("os_error_model.pt");
            }

            let os_error_model_data = web::Data::new(Mutex::new((
                (
                    Box::new(error_model) as Box<dyn nn::Module + Send>,
                    Box::new(os_model) as Box<dyn nn::Module + Send>,
                ),
                vs_os,
            )));

            HttpServer::new(move || {
                App::new()
                    .app_data(model_data.clone())
                    .app_data(os_error_model_data.clone())
                    .route("/predict", web::post().to(predict))
                    .route("/predict-os-error", web::post().to(predict_os_error))
                    .route(
                        "/",
                        web::get().to(|| async {
                            HttpResponse::Ok().body(
                                "Сервер классификации изображений запущен!\n\
                                              Используйте:\n\
                                              POST /predict - для общей классификации\n\
                                              POST /predict-os-error - для анализа ошибок ОС",
                            )
                        }),
                    )
            })
            .bind("0.0.0.0:5000")?
            .run()
            .await
        }
        Commands::Train => {
            println!("Обучение модели...");

            let device = Device::Cpu;
            let vs = nn::VarStore::new(device);
            let model = simple_cnn(&vs.root(), 10);

            let (train_images, train_labels) = create_dummy_data();
            train_model(&model, &train_images, &train_labels, &vs);

            Ok(())
        }
        Commands::Predict { model, image } => {
            match predict_from_cli(&model, &image) {
                Ok(_) => println!("Предсказание выполнено успешно"),
                Err(e) => eprintln!("Ошибка при предсказании: {}", e),
            }
            Ok(())
        }
        Commands::TrainOsError => {
            println!("Обучение модели для предсказания ошибок ОС...");
            match train_os_error_model() {
                Ok(_) => println!("Обучение завершено успешно"),
                Err(e) => eprintln!("Ошибка при обучении: {}", e),
            }
            Ok(())
        }
        Commands::PredictOsError { model, screenshot } => {
            match predict_os_error_from_cli(&model, &screenshot) {
                Ok(_) => println!("\nАнализ скриншота завершен успешно"),
                Err(e) => eprintln!("Ошибка при анализе скриншота: {}", e),
            }
            Ok(())
        }
    }
}
