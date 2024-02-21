# ZIP_PASSWORD=$(gopass cat private/zip-password) make data.tar.bz2.des3
.PHONY: data.tar.bz2.des3
data.tar.bz2.des3:
	tar -cjf data.tar.bz2 data
	openssl des3 -in data.tar.bz2 -out data.tar.bz2.des3 -k $(ZIP_PASSWORD)
	rm data.tar.bz2

.PHONY: get-data
get-data:
	openssl des3 -d -in data.tar.bz2.des3 -out data.tar.bz2 -k $(ZIP_PASSWORD)
	tar -xjf data.tar.bz2
