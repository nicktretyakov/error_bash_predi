
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Использование: $0 <путь_к_изображению>"
    exit 1
fi

echo "Компиляция проекта..."
cargo build --release

echo "Предсказание для изображения: $1"
./target/release/my-project predict --image "$1"
