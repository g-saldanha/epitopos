--
-- PostgreSQL database dump
--

-- Dumped from database version 11.3
-- Dumped by pg_dump version 11.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

DROP DATABASE IF EXISTS postgres;
--
-- Name: postgres; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE postgres WITH TEMPLATE = template0 ENCODING = 'UTF8' LC_COLLATE = 'English_United Kingdom.1252' LC_CTYPE = 'English_United Kingdom.1252';


ALTER DATABASE postgres OWNER TO postgres;

\connect postgres

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE postgres; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE postgres IS 'default administrative connection database';


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: epitopos_full; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.epitopos_full (
    attributes character varying,
    movelets integer,
    class character varying
);


ALTER TABLE public.epitopos_full OWNER TO postgres;

--
-- Data for Name: epitopos_full; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[hidrofobicidade]', 1, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[polaridade]', 5, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, superficie, antigenicidade, hidrofobicidade, polaridade]', 1, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred]', 7798, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[pi, antigenicidade, polaridade]', 1, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, superficie, antigenicidade, polaridade]', 1, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, superficie, antigenicidade, hidrofobicidade]', 3, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[pi, hidrofobicidade]', 2, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[hidrofobicidade, polaridade]', 6, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[antigenicidade, hidrofobicidade, polaridade]', 1, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[pi, hidrofobicidade, polaridade]', 5, 'bom');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, superficie, antigenicidade, polaridade]', 3, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, antigenicidade, polaridade]', 12, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[polaridade]', 54, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred]', 8309, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[pi]', 5, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, antigenicidade, hidrofobicidade, polaridade]', 4, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, polaridade]', 25, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, antigenicidade, polaridade]', 2, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, superficie, polaridade]', 3, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[superficie, antigenicidade, hidrofobicidade, polaridade]', 5, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, pi, superficie, antigenicidade, polaridade]', 10, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[aminoacido, bepipred, superficie, antigenicidade, hidrofobicidade, polaridade]', 15, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[hidrofobicidade, polaridade]', 2, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, polaridade]', 9, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[antigenicidade, hidrofobicidade, polaridade]', 4, 'ruim');
INSERT INTO public.epitopos_full (attributes, movelets, class) VALUES ('[bepipred, hidrofobicidade, polaridade]', 6, 'ruim');


--
-- PostgreSQL database dump complete
--

