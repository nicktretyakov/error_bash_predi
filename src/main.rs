
use actix_web::{web, App, HttpResponse, HttpServer, Responder, Result, HttpRequest};
use actix_web_actors::ws;
use actix::{Actor, StreamHandler, Handler, Message, Addr};
use tch::{nn, nn::Module, Device, Tensor, Kind};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::collections::HashMap;
use clap::{Parser, Subcommand};
use image::GenericImageView;
use std::path::Path;
use uuid::Uuid;
use base64;

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

// Структуры для чата
#[derive(Deserialize)]
struct ChatMessage {
    message: String,
    image_data: Option<String>, // Base64 encoded image
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    analysis: Option<ErrorAnalysis>,
    suggestions: Vec<String>,
}

#[derive(Serialize)]
struct ErrorAnalysis {
    error_type: String,
    os_type: String,
    confidence: f32,
    detailed_description: String,
    possible_causes: Vec<String>,
    solutions: Vec<String>,
}

// Типы ошибок операционных систем
const OS_ERROR_TYPES: &[&str] = &[
    "blue_screen_of_death",      // Синий экран смерти Windows
    "kernel_panic",              // Паника ядра Linux/macOS
    "application_crash",         // Краш приложения
    "memory_error",              // Ошибка памяти
    "disk_error",                // Ошибка диска
    "network_error",             // Сетевая ошибка
    "permission_denied",         // Отказ в доступе
    "file_not_found",           // Файл не найден
    "system_overload",          // Перегрузка системы
    "driver_error"              // Ошибка драйвера
];

const OS_TYPES: &[&str] = &["windows", "linux", "macos", "unknown"];

// WebSocket актор для чата
struct ChatSession {
    id: Uuid,
    addr: Addr<ChatServer>,
}

impl Actor for ChatSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.addr.do_send(Connect {
            id: self.id,
            addr: addr.recipient(),
        });
    }

    fn stopping(&mut self, _: &mut Self::Context) -> actix::Running {
        self.addr.do_send(Disconnect { id: self.id });
        actix::Running::Stop
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for ChatSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                if let Ok(chat_msg) = serde_json::from_str::<ChatMessage>(&text) {
                    self.addr.do_send(ClientMessage {
                        id: self.id,
                        msg: chat_msg,
                        addr: ctx.address().recipient(),
                    });
                }
            }
            Ok(ws::Message::Binary(_)) => println!("Unexpected binary"),
            _ => (),
        }
    }
}

// Сервер чата
struct ChatServer {
    sessions: HashMap<Uuid, actix::Recipient<ws::Message>>,
    models: std::sync::Arc<Mutex<((Box<dyn nn::Module + Send>, Box<dyn nn::Module + Send>), nn::VarStore)>>,
}

impl ChatServer {
    fn new(models: std::sync::Arc<Mutex<((Box<dyn nn::Module + Send>, Box<dyn nn::Module + Send>), nn::VarStore)>>) -> Self {
        ChatServer {
            sessions: HashMap::new(),
            models,
        }
    }
}

impl Actor for ChatServer {
    type Context = actix::Context<Self>;
}

// Сообщения для актора
#[derive(Message)]
#[rtype(result = "()")]
struct Connect {
    id: Uuid,
    addr: actix::Recipient<ws::Message>,
}

#[derive(Message)]
#[rtype(result = "()")]
struct Disconnect {
    id: Uuid,
}

#[derive(Message)]
#[rtype(result = "()")]
struct ClientMessage {
    id: Uuid,
    msg: ChatMessage,
    addr: actix::Recipient<ws::Message>,
}

impl Handler<Connect> for ChatServer {
    type Result = ();

    fn handle(&mut self, msg: Connect, _: &mut Self::Context) {
        self.sessions.insert(msg.id, msg.addr);
    }
}

impl Handler<Disconnect> for ChatServer {
    type Result = ();

    fn handle(&mut self, msg: Disconnect, _: &mut Self::Context) {
        self.sessions.remove(&msg.id);
    }
}

impl Handler<ClientMessage> for ChatServer {
    type Result = ();

    fn handle(&mut self, msg: ClientMessage, _: &mut Self::Context) {
        let response = self.process_chat_message(&msg.msg);
        let response_json = serde_json::to_string(&response).unwrap();

        if let Some(addr) = self.sessions.get(&msg.id) {
            let _ = addr.do_send(ws::Message::Text(response_json.into()));
        }
    }
}

impl ChatServer {
    fn process_chat_message(&self, msg: &ChatMessage) -> ChatResponse {
        if let Some(image_data) = &msg.image_data {
            // Обработка изображения
            if let Ok(analysis) = self.analyze_screenshot(image_data) {
                let suggestions = self.generate_suggestions(&analysis);

                ChatResponse {
                    response: format!("Я проанализировал ваш скриншот. Обнаружена ошибка типа '{}' в системе {}.",
                                    analysis.error_type, analysis.os_type),
                    analysis: Some(analysis),
                    suggestions,
                }
            } else {
                ChatResponse {
                    response: "Не удалось проанализировать изображение. Убедитесь, что это скриншот с ошибкой.".to_string(),
                    analysis: None,
                    suggestions: vec![
                        "Загрузите четкий скриншот ошибки".to_string(),
                        "Убедитесь, что изображение содержит текст ошибки".to_string(),
                    ],
                }
            }
        } else {
            // Обработка текстового сообщения
            self.process_text_query(&msg.message)
        }
    }

    fn analyze_screenshot(&self, image_data: &str) -> Result<ErrorAnalysis, Box<dyn std::error::Error>> {
        // Декодирование base64 изображения
        let image_bytes = base64::decode(image_data)?;
        let img = image::load_from_memory(&image_bytes)?;
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

        let image_tensor = Tensor::of_slice(&flat).view([1, 3, 128, 128]).to_device(Device::Cpu);

        let ((error_model, os_model), _) = &*self.models.lock().unwrap();

        let error_output = error_model.forward(&image_tensor);
        let error_probs = error_output.softmax(-1, Kind::Float);
        let (error_confidence, error_class) = error_probs.max_dim(-1, false);

        let os_output = os_model.forward(&image_tensor);
        let os_probs = os_output.softmax(-1, Kind::Float);
        let (_, os_class) = os_probs.max_dim(-1, false);

        let error_idx = i64::from(&error_class.get(0)) as usize;
        let os_idx = i64::from(&os_class.get(0)) as usize;

        let error_type = OS_ERROR_TYPES.get(error_idx).unwrap_or(&"unknown").to_string();
        let os_type = OS_TYPES.get(os_idx).unwrap_or(&"unknown").to_string();

        let (detailed_description, possible_causes, solutions) = self.get_detailed_error_info(&error_type, &os_type);

        Ok(ErrorAnalysis {
            error_type: error_type.clone(),
            os_type: os_type.clone(),
            confidence: f32::from(&error_confidence.get(0)),
            detailed_description,
            possible_causes,
            solutions,
        })
    }

    fn get_detailed_error_info(&self, error_type: &str, os_type: &str) -> (String, Vec<String>, Vec<String>) {
        match error_type {
            "blue_screen_of_death" => (
                "Синий экран смерти (BSOD) - критическая системная ошибка Windows, при которой операционная система не может продолжить работу и принудительно перезагружается.".to_string(),
                vec![
                    "Неисправность оборудования (RAM, жесткий диск)".to_string(),
                    "Несовместимые или поврежденные драйверы".to_string(),
                    "Перегрев процессора или видеокарты".to_string(),
                    "Поврежденные системные файлы".to_string(),
                ],
                vec![
                    "Проверьте код ошибки на экране и найдите его в документации Microsoft".to_string(),
                    "Запустите проверку памяти Windows (mdsched.exe)".to_string(),
                    "Обновите или откатите драйверы устройств".to_string(),
                    "Запустите sfc /scannow для проверки системных файлов".to_string(),
                    "Проверьте температуру компонентов".to_string(),
                ]
            ),
            "kernel_panic" => (
                "Паника ядра - критическая ошибка в ядре операционной системы Linux/macOS, после которой система не может продолжить безопасную работу.".to_string(),
                vec![
                    "Ошибки в модулях ядра".to_string(),
                    "Неисправность оборудования".to_string(),
                    "Несовместимые драйверы".to_string(),
                    "Переполнение стека ядра".to_string(),
                ],
                vec![
                    "Проанализируйте журналы системы (dmesg, /var/log/kern.log)".to_string(),
                    "Загрузитесь с предыдущего стабильного ядра".to_string(),
                    "Отключите недавно установленные модули".to_string(),
                    "Проверьте оборудование с помощью memtest86+".to_string(),
                ]
            ),
            "memory_error" => (
                "Ошибка памяти указывает на проблемы с доступом к оперативной памяти или её нехватку.".to_string(),
                vec![
                    "Физическая неисправность модулей RAM".to_string(),
                    "Нехватка оперативной памяти".to_string(),
                    "Ошибки в управлении памятью приложением".to_string(),
                ],
                vec![
                    "Запустите тест памяти (MemTest86, Windows Memory Diagnostic)".to_string(),
                    "Закройте ненужные приложения".to_string(),
                    "Увеличьте размер файла подкачки".to_string(),
                    "Переустановите или замените модули RAM".to_string(),
                ]
            ),
            _ => (
                "Общая системная ошибка, требующая дополнительной диагностики.".to_string(),
                vec!["Различные факторы могут вызывать эту ошибку".to_string()],
                vec![
                    "Перезагрузите систему".to_string(),
                    "Проверьте журналы событий".to_string(),
                    "Обратитесь к документации системы".to_string(),
                ]
            )
        }
    }

    fn process_text_query(&self, message: &str) -> ChatResponse {
        let message_lower = message.to_lowercase();

        if message_lower.contains("помощь") || message_lower.contains("help") {
            ChatResponse {
                response: "Я помогу вам диагностировать ошибки операционной системы! Загрузите скриншот ошибки, и я проанализирую её тип, определю ОС и предложу решения.".to_string(),
                analysis: None,
                suggestions: vec![
                    "Загрузите скриншот с ошибкой для анализа".to_string(),
                    "Спросите о конкретном типе ошибки".to_string(),
                    "Опишите симптомы проблемы".to_string(),
                ],
            }
        } else if message_lower.contains("bsod") || message_lower.contains("синий экран") {
            ChatResponse {
                response: "BSOD (Blue Screen of Death) - критическая ошибка Windows. Обычно вызвана проблемами с драйверами, оборудованием или системными файлами.".to_string(),
                analysis: None,
                suggestions: vec![
                    "Запишите код ошибки с синего экрана".to_string(),
                    "Проверьте последние установленные драйверы".to_string(),
                    "Запустите тест памяти".to_string(),
                ],
            }
        } else if message_lower.contains("kernel panic") || message_lower.contains("паника ядра") {
            ChatResponse {
                response: "Kernel Panic - критическая ошибка ядра в Linux/macOS. Система не может продолжить работу и перезагружается.".to_string(),
                analysis: None,
                suggestions: vec![
                    "Проверьте журналы системы (/var/log/kern.log)".to_string(),
                    "Загрузитесь с предыдущего ядра".to_string(),
                    "Отключите проблемные модули ядра".to_string(),
                ],
            }
        } else {
            ChatResponse {
                response: "Опишите вашу проблему подробнее или загрузите скриншот ошибки для анализа.".to_string(),
                analysis: None,
                suggestions: vec![
                    "Загрузите скриншот ошибки".to_string(),
                    "Укажите тип операционной системы".to_string(),
                    "Опишите когда возникла ошибка".to_string(),
                ],
            }
        }
    }

    fn generate_suggestions(&self, analysis: &ErrorAnalysis) -> Vec<String> {
        let mut suggestions = vec![
            "Сохраните скриншот ошибки для дальнейшего анализа".to_string(),
            "Запишите код ошибки, если он есть".to_string(),
        ];

        suggestions.extend(analysis.solutions.clone());
        suggestions
    }
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

// Специализированная модель CNN для анализа ошибок ОС с более крупными изображениями
fn os_error_cnn(p: &nn::Path, num_error_types: i64, num_os_types: i64) -> (impl nn::Module, impl nn::Module) {
    // Общие сверточные слои для извлечения признаков
    let conv1 = nn::conv2d(p / "conv1", 3, 64, 5, nn::ConvConfig { padding: 2, ..Default::default() });
    let conv2 = nn::conv2d(p / "conv2", 64, 128, 5, nn::ConvConfig { padding: 2, ..Default::default() });
    let conv3 = nn::conv2d(p / "conv3", 128, 256, 3, nn::ConvConfig { padding: 1, ..Default::default() });

    let shared_features = nn::seq()
        .add(conv1)
        .add_fn(|xs| xs.relu().max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false))
        .add(conv2)
        .add_fn(|xs| xs.relu().max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false))
        .add(conv3)
        .add_fn(|xs| xs.relu().max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false))
        .add_fn(|xs| xs.flatten(1, -1));

    // Классификатор типа ошибки
    let error_classifier = nn::seq()
        .add(nn::linear(p / "error_fc1", 256 * 16 * 16, 512, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::dropout(p / "error_dropout", 0.5))
        .add(nn::linear(p / "error_fc2", 512, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "error_out", 256, num_error_types, Default::default()));

    // Классификатор типа ОС
    let os_classifier = nn::seq()
        .add(nn::linear(p / "os_fc1", 256 * 16 * 16, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::dropout(p / "os_dropout", 0.3))
        .add(nn::linear(p / "os_out", 256, num_os_types, Default::default()));

    (
        nn::seq().add(shared_features).add(error_classifier),
        nn::seq().add_fn(move |xs| {
            let features = xs.apply(&shared_features);
            features.apply(&os_classifier)
        })
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

// Веб-обработчик для предсказания ошибок ОС
async fn predict_os_error(
    req: web::Json<PredictRequest>,
    model_data: web::Data<Mutex<((Box<dyn nn::Module + Send>, Box<dyn nn::Module + Send>), nn::VarStore)>>,
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

    let error_type = OS_ERROR_TYPES.get(error_idx).unwrap_or(&"unknown").to_string();
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
        _ => "Неизвестная ошибка"
    }.to_string();

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

    Ok(Tensor::of_slice(&flat).view([1, 3, 128, 128]).to_device(Device::Cpu))
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
    let (error_model, os_model) = os_error_cnn(&vs.root(), OS_ERROR_TYPES.len() as i64, OS_TYPES.len() as i64);

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
            println!("Epoch: {}, Error Loss: {:.4}, OS Loss: {:.4}, Total Loss: {:.4}",
                epoch, f64::from(&error_loss), f64::from(&os_loss), f64::from(&total_loss));
        }
    }

    vs.save("os_error_model.pt")?;
    println!("Модель для предсказания ошибок ОС сохранена в os_error_model.pt");
    Ok(())
}

// Функция для предсказания ошибок ОС
fn predict_os_error_from_cli(model_path: &str, screenshot_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let (error_model, os_model) = os_error_cnn(&vs.root(), OS_ERROR_TYPES.len() as i64, OS_TYPES.len() as i64);

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
        _ => "Неизвестная ошибка"
    };

    println!("=== Анализ ошибки операционной системы ===");
    println!("Тип ошибки: {}", error_type);
    println!("Операционная система: {}", os_type);
    println!("Уверенность (ошибка): {:.2}%", f32::from(&error_confidence.get(0)) * 100.0);
    println!("Уверенность (ОС): {:.2}%", f32::from(&os_confidence.get(0)) * 100.0);
    println!("Описание: {}", description);

    Ok(())
}

// WebSocket обработчик
async fn websocket_handler(
    req: HttpRequest,
    stream: web::Payload,
    srv: web::Data<Addr<ChatServer>>,
) -> Result<HttpResponse, actix_web::Error> {
    let chat_session = ChatSession {
        id: Uuid::new_v4(),
        addr: srv.get_ref().clone(),
    };

    ws::start(chat_session, &req, stream)
}

// Страница чата
async fn chat_page() -> Result<HttpResponse, actix_web::Error> {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>AI Error Analysis Chat</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: #007bff; color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }
        .chat-area { height: 400px; overflow-y: auto; padding: 20px; border-bottom: 1px solid #eee; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background: #e3f2fd; margin-left: 20%; }
        .bot-message { background: #f1f8e9; margin-right: 20%; }
        .error-analysis { background: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 10px 0; }
        .suggestions { background: #f3e5f5; border-left: 4px solid #9c27b0; padding: 15px; margin: 10px 0; }
        .input-area { padding: 20px; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .input-area button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .file-input { margin: 10px 0; }
        .status { padding: 5px 20px; background: #e8f5e8; border-bottom: 1px solid #ddd; font-size: 14px; }
        .error-type { font-weight: bold; color: #d32f2f; }
        .os-type { font-weight: bold; color: #1976d2; }
        .confidence { font-weight: bold; color: #388e3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Assistant для анализа ошибок ОС</h1>
            <p>Загрузите скриншот ошибки или задайте вопрос</p>
        </div>
        <div class="status" id="status">Подключение к серверу...</div>
        <div class="chat-area" id="chatArea"></div>
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Введите ваш вопрос или опишите проблему..." onkeypress="handleKeyPress(event)">
            <input type="file" id="imageInput" accept="image/*" class="file-input" onchange="handleImageUpload(event)">
            <button onclick="sendMessage()">Отправить</button>
        </div>
    </div>

    <script>
        let socket;
        let connected = false;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws/';

            socket = new WebSocket(wsUrl);

            socket.onopen = function(event) {
                connected = true;
                document.getElementById('status').textContent = '✅ Подключено к AI помощнику';
                document.getElementById('status').style.background = '#e8f5e8';

                addMessage('bot', 'Привет! Я AI помощник для анализа ошибок операционных систем. Загрузите скриншот ошибки или задайте вопрос о проблемах с компьютером.');
            };

            socket.onmessage = function(event) {
                const response = JSON.parse(event.data);
                handleBotResponse(response);
            };

            socket.onclose = function(event) {
                connected = false;
                document.getElementById('status').textContent = '❌ Соединение потеряно. Переподключение...';
                document.getElementById('status').style.background = '#ffebee';
                setTimeout(connectWebSocket, 3000);
            };

            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
                document.getElementById('status').textContent = '❌ Ошибка подключения';
                document.getElementById('status').style.background = '#ffebee';
            };
        }

        function addMessage(type, content, analysis = null, suggestions = null) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + type + '-message';

            let html = '<div>' + content + '</div>';

            if (analysis) {
                html += `
                    <div class="error-analysis">
                        <h4>📊 Анализ ошибки:</h4>
                        <p><span class="error-type">Тип ошибки:</span> ${analysis.error_type}</p>
                        <p><span class="os-type">Операционная система:</span> ${analysis.os_type}</p>
                        <p><span class="confidence">Уверенность:</span> ${(analysis.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Описание:</strong> ${analysis.detailed_description}</p>

                        <h5>🔍 Возможные причины:</h5>
                        <ul>
                            ${analysis.possible_causes.map(cause => `<li>${cause}</li>`).join('')}
                        </ul>

                        <h5>🛠️ Рекомендуемые решения:</h5>
                        <ul>
                            ${analysis.solutions.map(solution => `<li>${solution}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            if (suggestions && suggestions.length > 0) {
                html += `
                    <div class="suggestions">
                        <h5>💡 Дополнительные рекомендации:</h5>
                        <ul>
                            ${suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            messageDiv.innerHTML = html;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function handleBotResponse(response) {
            addMessage('bot', response.response, response.analysis, response.suggestions);
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (message && connected) {
                addMessage('user', message);

                socket.send(JSON.stringify({
                    message: message,
                    image_data: null
                }));

                input.value = '';
            }
        }

        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (file && connected) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const base64Data = e.target.result.split(',')[1];

                    addMessage('user', '📷 Скриншот загружен для анализа...');

                    socket.send(JSON.stringify({
                        message: "Проанализируйте этот скриншот с ошибкой",
                        image_data: base64Data
                    }));
                };
                reader.readAsDataURL(file);

                // Очистить input
                event.target.value = '';
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        // Подключение при загрузке страницы
        connectWebSocket();
    </script>
</body>
</html>
    "#;

    Ok(HttpResponse::Ok().content_type("text/html").body(html))
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

            // Инициализация модели для ошибок ОС
            let device = Device::Cpu;
            let vs_os = nn::VarStore::new(device);
            let (error_model, os_model) = os_error_cnn(&vs_os.root(), OS_ERROR_TYPES.len() as i64, OS_TYPES.len() as i64);

            // Попытка загрузить модель для ошибок ОС
            if vs_os.load("os_error_model.pt").is_err() {
                println!("Модель для ошибок ОС не найдена, создание новой...");
                match train_os_error_model() {
                    Ok(_) => println!("Модель для ошибок ОС создана успешно"),
                    Err(e) => println!("Ошибка создания модели для ошибок ОС: {}", e),
                }
                let _ = vs_os.load("os_error_model.pt");
            }

            let os_error_model_data = web::Data::new(Mutex::new(
                ((Box::new(error_model) as Box<dyn nn::Module + Send>,
                  Box::new(os_model) as Box<dyn nn::Module + Send>), vs_os)
            ));

            // Создание сервера чата
            let chat_server = ChatServer::new(os_error_model_data.clone().into_inner()).start();

            HttpServer::new(move || {
                App::new()
                    .app_data(model_data.clone())
                    .app_data(os_error_model_data.clone())
                    .app_data(web::Data::new(chat_server.clone()))
                    .route("/predict", web::post().to(predict))
                    .route("/predict-os-error", web::post().to(predict_os_error))
                    .route("/ws/", web::get().to(websocket_handler))
                    .route("/chat", web::get().to(chat_page))
                    .route("/", web::get().to(|| async {
                        HttpResponse::Ok().body("Сервер классификации изображений запущен!\n\
                                              Используйте:\n\
                                              POST /predict - для общей классификации\n\
                                              POST /predict-os-error - для анализа ошибок ОС\n\
                                              GET /chat - для чата с AI помощником\n\
                                              WS /ws/ - WebSocket подключение для чата")
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
        },
        Commands::TrainOsError => {
            println!("Обучение модели для предсказания ошибок ОС...");
            match train_os_error_model() {
                Ok(_) => println!("Обучение завершено успешно"),
                Err(e) => eprintln!("Ошибка при обучении: {}", e),
            }
            Ok(())
        },
        Commands::PredictOsError { model, screenshot } => {
            match predict_os_error_from_cli(&model, &screenshot) {
                Ok(_) => println!("\nАнализ скриншота завершен успешно"),
                Err(e) => eprintln!("Ошибка при анализе скриншота: {}", e),
            }
            Ok(())
        }
    }
}
