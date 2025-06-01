
use actix_web::{web, App, HttpResponse, HttpServer, Responder, Result};
use tch::{nn, nn::Module, Device, Tensor, Kind};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use clap::{Parser, Subcommand};
use image::GenericImageView;
use std::path::Path;

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

// Упрощенная модель CNN для демонстрации
fn simple_cnn(p: &nn::Path, num_classes: i64) -> impl nn::Module {
    let conv1 = nn::conv2d(p / "conv1", 3, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() });
    let conv2 = nn::conv2d(p / "conv2", 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() });
    let fc1 = nn::linear(p / "fc1", 64 * 8 * 8, 128, Default::default());
    let fc2 = nn::linear(p / "fc2", 128, num_classes, Default::default());

    nn::seq()
        .add(conv1)
        .add_fn(|xs| xs.relu().max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false))
        .add(conv2)
        .add_fn(|xs| xs.relu().max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false))
        .add_fn(|xs| xs.flatten(1, -1))
        .add(fc1)
        .add_fn(|xs| xs.relu())
        .add(fc2)
}

// Создание тестовых данных для демонстрации
fn create_dummy_data() -> (Tensor, Tensor) {
    let device = Device::Cpu;
    let train_images = Tensor::randn(&[100, 3, 32, 32], (Kind::Float, device));
    let train_labels = Tensor::randint(10, &[100], (Kind::Int64, device));
    (train_images, train_labels)
}

// Обучение модели
fn train_model(model: &dyn nn::Module, train_images: &Tensor, train_labels: &Tensor, vs: &nn::VarStore) {
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

    Ok(Tensor::of_slice(&flat).view([1, 3, 32, 32]).to_device(Device::Cpu))
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

            let model_data = web::Data::new(Mutex::new((Box::new(model) as Box<dyn nn::Module + Send>, vs)));

            HttpServer::new(move || {
                App::new()
                    .app_data(model_data.clone())
                    .route("/predict", web::post().to(predict))
                    .route("/", web::get().to(|| async {
                        HttpResponse::Ok().body("Сервер классификации изображений запущен! Используйте POST /predict")
                    }))
            })
            .bind("0.0.0.0:5000")?
            .run()
            .await
        },
        Commands::Train => {
            println!("Обучение модели...");

            let device = Device::Cpu;
            let vs = nn::VarStore::new(device);
            let model = simple_cnn(&vs.root(), 10);

            let (train_images, train_labels) = create_dummy_data();
            train_model(&model, &train_images, &train_labels, &vs);

            Ok(())
        },
        Commands::Predict { model, image } => {
            match predict_from_cli(&model, &image) {
                Ok(_) => println!("Предсказание выполнено успешно"),
                Err(e) => eprintln!("Ошибка при предсказании: {}", e),
            }
            Ok(())
        }
    }
}
