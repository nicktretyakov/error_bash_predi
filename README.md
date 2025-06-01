# Image Classification CLI Tool

A Rust-based command-line utility and web server for image classification using machine learning. This project provides both a web API and CLI interface for predicting image classes using a simple Convolutional Neural Network (CNN).

## Features

- **CLI Interface**: Predict image classes directly from the command line
- **Web Server**: REST API for image classification
- **Model Training**: Train your own classification models
- **Multiple Formats**: Support for various image formats
- **Rust Performance**: Built with Rust for high performance and safety

## Dependencies

This project uses the following main dependencies:
- `tch` - PyTorch bindings for Rust
- `actix-web` - Web framework for the REST API
- `clap` - Command-line argument parsing
- `image` - Image processing library
- `serde` - Serialization/deserialization

## Installation

1. Clone this repository
2. Ensure you have Rust installed (https://rustup.rs/)
3. Build the project:
   ```bash
   cargo build --release
   ```

## Usage

### Command Line Interface

The tool provides three main commands:

#### 1. Train a Model
```bash
cargo run train
```
or using the convenience script:
```bash
./train_model.sh
```

#### 2. Start Web Server
```bash
cargo run server
```
or using the convenience script:
```bash
./run_server.sh
```
The server will start on `http://0.0.0.0:5000`

#### 3. Predict Image Class
```bash
cargo run predict --image path/to/your/image.jpg
```
or using the convenience script:
```bash
./predict_image.sh path/to/your/image.jpg
```

### Web API

Once the server is running, you can make predictions via HTTP POST requests:

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "image": [/* flattened image array as f32 values */]
}
```

**Response**:
```json
{
  "class": 5,
  "confidence": 0.87
}
```

### Example Usage

1. First, train a model:
   ```bash
   ./train_model.sh
   ```

2. Predict an image class:
   ```bash
   ./predict_image.sh sample_image.jpg
   ```

3. Or start the web server:
   ```bash
   ./run_server.sh
   ```

## Model Architecture

The project uses a simple CNN architecture with:
- Two convolutional layers (3→32→64 channels)
- ReLU activation functions
- Max pooling layers
- Two fully connected layers
- Designed for 32x32 RGB images

## File Structure

- `src/main.rs` - Main application code
- `Cargo.toml` - Project dependencies and metadata
- `*.sh` - Convenience scripts for common operations
- `model.pt` - Saved model file (generated after training)

## Configuration

### Image Requirements
- Images are automatically resized to 32x32 pixels
- Supports common formats (JPEG, PNG, etc.)
- RGB color images

### Model Parameters
- Input size: 32x32x3 (RGB)
- Number of classes: 10 (configurable)
- Learning rate: 1e-3
- Training epochs: 5 (default)

## Development

### Building
```bash
cargo build
```

### Running Tests
```bash
cargo test
```

### Development Server
```bash
cargo run server
```

## Troubleshooting

### Common Issues

1. **Missing OpenSSL**: Install development packages
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libssl-dev

   # Fedora
   sudo dnf install openssl-devel
   ```

2. **Model Not Found**: Train a model first using `cargo run train`

3. **Image Format Issues**: Ensure your image is in a supported format (JPEG, PNG, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please refer to the LICENSE file for details.

## Performance Notes

- The model uses CPU by default for better compatibility
- For GPU acceleration, ensure CUDA is properly configured with PyTorch
- Image preprocessing is optimized for batch operations

## Future Enhancements

- Support for larger image sizes
- Pre-trained model integration
- Additional CNN architectures
- Batch prediction API
- Model evaluation metrics
