NAME=

$(NAME): $(NAME).cpp
	g++ -std=gnu++11 -O3 -o $@ $<

debug: $(NAME).cpp
	g++ -fsanitize=address -std=gnu++11 -O0 -g -o $@ $<

zip: $(NAME).cpp
	mkdir -p submissions
	zip `date +submissions/%m%d_%H%M%S.zip` $<
