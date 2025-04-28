CREATE TABLE dim_time (
    time_id SERIAL PRIMARY KEY,
    date_time TIMESTAMP,
    date_documented DATE,
    year INT,
    month INT,
    hour INT,
    season VARCHAR(20)
);


CREATE TABLE dim_location (
    location_id SERIAL PRIMARY KEY,
    country_code VARCHAR(10),
    country VARCHAR(100),
    region VARCHAR(100),
    locale VARCHAR(100),
    latitude DOUBLE,
    longitude DOUBLE
);


CREATE TABLE dim_ufo (
    ufo_id SERIAL PRIMARY KEY,
    ufo_shape VARCHAR(50)
);


CREATE TABLE fact_sightings (
    sighting_id SERIAL PRIMARY KEY,
    time_id INT REFERENCES dim_time(time_id),
    location_id INT REFERENCES dim_location(location_id),
    ufo_id INT REFERENCES dim_ufo(ufo_id),
    length_of_encounter_seconds BIGINT,
    encounter_duration VARCHAR(50),
    description TEXT
);

CREATE SEQUENCE time_id_seq START 1;
CREATE TABLE dim_time (
    time_id INTEGER DEFAULT nextval('time_id_seq') PRIMARY KEY,
    date_time TIMESTAMP,
    date_documented DATE,
    year INT,
    month INT,
    hour INT,
    season VARCHAR(20)
);




INSERT INTO dim_time (date_time, date_documented, year, month, hour, season)
SELECT DISTINCT Date_time, date_documented, Year, Month, Hour, Season
FROM sightings;


INSERT INTO dim_location (country_code, country, region, locale, latitude, longitude)
SELECT DISTINCT Country_Code, Country, Region, Locale, latitude, longitude
FROM sightings;


INSERT INTO dim_ufo (ufo_shape)
SELECT DISTINCT UFO_shape
FROM sightings;



INSERT INTO fact_sightings (time_id, location_id, ufo_id, length_of_encounter_seconds, encounter_duration, description)
SELECT 
    t.time_id,
    l.location_id,
    u.ufo_id,
    s.length_of_encounter_seconds,
    s.Encounter_Duration,
    s.Description
FROM sightings s
JOIN dim_time t ON s.Date_time = t.date_time 
                AND s.date_documented = t.date_documented 
                AND s.Year = t.year 
                AND s.Month = t.month 
                AND s.Hour = t.hour 
                AND s.Season = t.season
JOIN dim_location l ON s.Country_Code = l.country_code 
                     AND s.Country = l.country 
                     AND s.Region = l.region 
                     AND s.Locale = l.locale 
                     AND s.latitude = l.latitude 
                     AND s.longitude = l.longitude
JOIN dim_ufo u ON s.UFO_shape = u.ufo_shape;
