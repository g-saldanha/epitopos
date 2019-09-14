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
-- Name: epitopos_without_bepipred; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.epitopos_without_bepipred (
    attributes character varying,
    movelets integer,
    class character varying
);


ALTER TABLE public.epitopos_without_bepipred OWNER TO postgres;

--
-- Data for Name: epitopos_without_bepipred; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[superficie, antigenicidade, hidrofobicidade, polaridade]', 1622, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[superficie, polaridade]', 92, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, superficie, hidrofobicidade, polaridade]', 196, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, antigenicidade, hidrofobicidade, polaridade]', 165, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[aminoacido, pi, superficie, polaridade]', 150, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, polaridade]', 3, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[superficie, antigenicidade, polaridade]', 1425, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[antigenicidade, hidrofobicidade, polaridade]', 209, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[antigenicidade, polaridade]', 301, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, superficie, antigenicidade, polaridade]', 5, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, superficie, antigenicidade, hidrofobicidade, polaridade]', 638, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, superficie, polaridade]', 6, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[hidrofobicidade, polaridade]', 115, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[aminoacido, pi, superficie, hidrofobicidade, polaridade]', 12, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi]', 188, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, antigenicidade, polaridade]', 52, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[aminoacido]', 2140, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[superficie, hidrofobicidade, polaridade]', 4, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[polaridade]', 4097, 'ruim');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, hidrofobicidade]', 267, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[hidrofobicidade, polaridade]', 1827, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[hidrofobicidade]', 522, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[superficie]', 513, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, antigenicidade, polaridade]', 314, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[aminoacido]', 2809, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[antigenicidade, hidrofobicidade, polaridade]', 110, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[polaridade]', 2491, 'bom');
INSERT INTO public.epitopos_without_bepipred (attributes, movelets, class) VALUES ('[pi, hidrofobicidade, polaridade]', 2084, 'bom');


--
-- PostgreSQL database dump complete
--

