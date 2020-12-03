# Recognition of handwritten (long) numbers
> Neural Network recognizer of long handwritten numbers via the use of a webcam.

The aim of this project is to reach a good level of accuracy in the recognition of long numbers (composed of several digits) written by hand. 
![](header.png)

## Download

Linux/MacOS/Windows:

```sh
git clone https://github.com/filippoguerranti/handwritten_long_digit_recognition.git
cd handwritten_long_number_recognition
```

## Development setup

After downloading the directory, make sure all the dependencies are installed.
In particular: 

```sh
numpy, torch, torchvision, opencv, matplotlib
```

Install them if needed. To check if everything is correctly set run the following command:

```sh
python3 requirements_check.py
```

You should get something like:

```sh
All the dependecies are correctly installed.
```

Otherwise, you will get an error message explaining you what to do.

## Usage example

In order to start the application, do the following

```sh
python3 main.py
```

_For more examples and usage, please refer to the [Wiki][wiki]._


## Release History

* 0.0.1
    * Work in progress

## Meta

Filippo Guerranti – filippo.guerranti@student.unisi.it

Distributed under the Apache-2.0 License. See ``LICENSE`` for more information.

[https://github.com/filippoguerranti/handwritten_long_digit_recognition](https://github.com/filippoguerranti/handwritten_long_digit_recognition)

## Contributing

1. Fork it (<https://github.com/filippoguerranti/handwritten_long_digit_recognition/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[wiki]: https://github.com/filippoguerranti/handwritten_long_digit_recognition/wiki
