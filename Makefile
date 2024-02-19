# ZIP_PASSWORD=$(gopass cat private/zip-password) make train.tar.bz2.des3
train.tar.bz2.des3: train-x train-y
	tar -cjf train.tar.bz2 train-x train-y
	openssl des3 -in train.tar.bz2 -out train.tar.bz2.des3 -k $(ZIP_PASSWORD)
	rm train.tar.bz2

get-train:
	openssl des3 -d -in train.tar.bz2.des3 -out train.tar.bz2 -k $(ZIP_PASSWORD)
	tar -xjf train.tar.bz2

