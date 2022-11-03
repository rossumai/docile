install:
	echo "Installing binary dependencies"
	brew install poetry poppler

	echo "Installing poetry dependencies"
	poetry install --with test --with dev --without doctr

	echo "Installing pre-commit"
	poetry run pre-commit install


install-with-ocr: install
	echo "Installing binary dependencies for docTR"
	brew install cairo pango gdk-pixbuf libffi

	echo "Installing poetry dependencies"
	poetry install --with test --with dev


# NOTE: Use this at your own risk
install-with-ocr-m1: install-with-ocr
	sudo ./m1_create_missing_symlinks.sh


test:
	poetry run pytest

jupyter:
	poetry run python3 -m jupyter notebook
