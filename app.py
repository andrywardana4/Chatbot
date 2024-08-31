from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai
import logging
import random

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

genai.configure(api_key="API_KEY")

generation_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro-001",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

prompt_parts = [
    # Masukkan prompt parts di sini seperti pada kode Anda
    "dinas: layanan",
    "penjelasan dinas perdangan: Perindustrian Verifikasi Ijin pada OSS, Validasi Cek Lokasi Usaha Industri, Surat Keterangan Pelaku Industri, Surat Pengantar Pengiriman Asal Barang, Pendaftaran Akun SIINas, Sertifikat TKDN. Perdagangan Perdagangan Luar Negeri & Perdagangan Dalam Negeri.Metrologi perizinan Alat - Alat Ukur, Takar, Timbang dan Perlengkapannya (UTTP)",
    "dinas: tentang kami",
    "penjelasan dinas perdangan: Website Resmi Dinas Perdagangan dan Perindustrian Kabupaten Kotawaringin Timur. Kantor Jalan Jenderal Sudirman 6,7 Sampit, Kalimantan Tengah Kode Pos 74322",
    "dinas: sosial media kami"
    "penjelasan dinas perdangan: website, dispedagin.kotimkab.go.id ", "instagram, disperdagin.kotim", "email, disperidag@kotimkab.go.id"
    "dinas: informasi publik",
    "penjelasan dinas perdangan: belom ada isi" "Kami menyediakan informasi publik terkait dengan layanan perizinan, kebijakan, dan program yang dijalankan oleh Dinas Perdagangan dan Perindustrian Kabupaten Kotawaringin Timur.",
    "dinas: harga daging",
    "penjelasan dinas perdangan: harga sapi bahan dalam pada tanggal 25 oktober tahun 2023 per kilo adalah 160000",
    "dinas: estimasi waktu pembuatan izin UMKM",
    "penjelasan_dinas_perdagangan : Estimasi waktu pembuatan izin UMKM adalah 1 minggu.",
    "dinas: estimasi waktu pembuatan izin Perdagangan Dalam Negeri",
    "penjelasan_dinas_perdagangan: Estimasi waktu pembuatan izin Perdagangan Dalam Negeri adalah 1-2 minggu.",
    "dinas: estimasi waktu pembuatan izin Industri",
    "penjelasan_dinas_perdagangan: Estimasi waktu pembuatan izin Industri adalah 2-3 minggu.",
    "dinas: estimasi waktu pembuatan izin Perdagangan Luar Negeri",
    "penjelasan_dinas_perdagangan: Estimasi waktu pembuatan izin Perdagangan Luar Negeri adalah 2 minggu.",
    "dinas: estimasi waktu pembuatan izin Metrologi",
    "penjelasan_dinas_perdagangan: Estimasi waktu pembuatan izin Metrologi untuk Alat - Alat Ukur, Takar, Timbang dan Perlengkapannya (UTTP) adalah 1 minggu."
    "dinas: kepala dinas",
    "penjelasan dinas perdangan: pemimpin dinas perdagangan kabupaten kotawaringin timur, Dr.Drs.H.zulhaidir,M.si. pembina tingkat 1(IV/b). NIP 191611161994031006",
    "dinas: sekretaris",
    "penjelasan dinas perdangan: mohhammad ikhwan,s.t.,mm. pembina (IV/a). NIP 198006082009041001",
    "dinas: kepala sub bagian umum dan pelaporan",
    "penjelasan dinas perdangan: kamtini,s.e. penata tingkat I(III/d) NIP 197207221999022001",
    "dinas: kepala sub bagian keuangan dan perencanaan",
    "penjelasan dinas perdangan: Aulia rahman, S,psi., M.A.P penata tingkat 1(III/d) NIP 198508132008041002",
    "dinas: kepala bidang perdagangan",
    "penjelasan dinas perdangan: KASIYAN, SE.,MM pembina(IV/a) NIP 197309201993031004",
    "dinas: kepala bidang metrologi legal",
    "penjelasan dinas perdangan: TAUBA s,sos.,msi pembina(IV/a) NIP 196811151995121004",
    "dinas: kepala bidang perindustrian",
    "penjelasan dinas perdangan: rodi hartono, s.e.,m.t pembina (IV/a) NIP 196901122000031008",
    "dinas: Bagaimana cara menghubunginya?",
    "penjelasan dinas perdangan:Telepon 0531 2118",
    "dinas: Alamat disperdagin kotim?",
    "penjelasana disperdagin:  Dinas Perdagangan dan Perindustrian Kab. Kotim Jl. Jendral Sudirman Km 6,7",
    "dinas: bagaimana cara mengurus izin usaha?",
    "Anda bisa mengurus izin usaha melalui layanan OSS atau langsung datang ke kantor kami.",
    "dinas: dimana kantor Dinas Perdagangan dan Perindustrian?",
    "Kantor kami berada di lokasi yang mudah diakses di pusat kota. Silakan cek website resmi untuk alamat terbaru.",
    "dinas: apa jam operasional kantor?",
    "Jam operasional kantor kami dari Senin hingga Jumat, pukul 08:00 hingga 16:00.",
    "dinas: bagaimana cara mengajukan keluhan?",
    "Anda bisa mengajukan keluhan melalui website resmi kami atau datang langsung ke kantor.",
    "dinas: apa saja syarat untuk izin impor?",
    "Syarat untuk izin impor termasuk NIB, SIUP, dan dokumen pendukung lainnya. Informasi lengkap bisa Anda dapatkan di kantor kami.",
    "dinas: berapa biaya untuk mengurus izin usaha?",
    "Biaya untuk mengurus izin usaha tergantung jenis usaha Anda. Silakan datang ke kantor kami untuk informasi lebih lanjut.",
    "dinas: apa itu program UMKM?",
    "Program UMKM adalah inisiatif untuk mendukung usaha mikro, kecil, dan menengah di Kabupaten Kotawaringin Timur.",
    "dinas: bagaimana cara mendaftar program UMKM?",
    "Anda bisa mendaftar program UMKM dengan mengisi formulir di kantor kami atau melalui website resmi kami.",
    "dinas: ada pelatihan untuk UMKM?",
    "Ya, kami sering mengadakan pelatihan untuk UMKM. Informasi jadwal pelatihan bisa Anda dapatkan di website resmi kami.",
    "dinas: bagaimana kondisi pasar tradisional?",
    "Pasar tradisional saat ini dalam kondisi baik dan selalu dipantau oleh dinas kami.",
    "dinas: ada informasi tentang harga bahan pokok?",
    "Informasi harga bahan pokok terbaru bisa Anda cek di website resmi Dinas Perdagangan dan Perindustrian Kabupaten Kotawaringin Timur.",
    "dinas: bagaimana cara mengajukan bantuan usaha?",
    "Untuk mengajukan bantuan usaha, Anda perlu mengisi formulir dan melampirkan dokumen pendukung. Formulir bisa diambil di kantor kami.",
    "dinas: apa saja produk unggulan daerah ini?",
    "Produk unggulan Kabupaten Kotawaringin Timur meliputi karet, rotan, dan hasil laut.",
    "dinas: bagaimana cara ikut pameran dagang?",
    "Untuk ikut pameran dagang, Anda bisa mendaftar melalui Dinas Perdagangan dan Perindustrian atau mengikuti pengumuman resmi di website kami.",
    "dinas: apa program unggulan dinas?",
    "Program unggulan kami termasuk pengembangan UMKM, pelatihan keterampilan, dan peningkatan akses pasar.",
    "dinas: ada lowongan kerja di dinas?",
    "Informasi lowongan kerja bisa Anda cek di papan pengumuman kantor atau di website resmi kami.",
    "dinas: dimana saya bisa melaporkan masalah perdagangan?",
    "Anda bisa melaporkan masalah perdagangan langsung ke kantor kami atau melalui call center dinas.",
    "dinas: bisakah saya konsultasi tentang industri kreatif?",
    "Tentu, kami menyediakan layanan konsultasi untuk industri kreatif. Silakan datang ke kantor kami untuk informasi lebih lanjut."
    "kontak", 
    "Telepon (0531) 2118, Email: disperindagkotim@example.com, Alamat: Jl. Jenderal Sudirman No. 45, Sampit, Kabupaten Kotawaringin Timur, Kalimantan Tengah.",
    "pelayanan_informasi_konsultasi", 
    "Informasi tentang peluang usaha, konsultasi tentang pengembangan usaha kecil dan menengah, serta informasi pasar dan pemasaran produk.",
    "pelayanan_pemberdayaan_pengembangan", 
    "Pelatihan dan bimbingan teknis bagi pelaku usaha, pembinaan terhadap industri kecil dan menengah (IKM), dan pengembangan sentra-sentra industri.",
    "pelayanan_pengaduan_pengawasan", 
    "Penerimaan dan penanganan pengaduan konsumen, pengawasan barang beredar dan jasa, serta pengawasan dan perlindungan terhadap konsumen.",
    "siup", 
    "SIUP Mikro, Kecil, Menengah, dan Besar. Persyaratan umum: Fotokopi KTP, NPWP, akta pendirian usaha (untuk badan usaha), dan dokumen pendukung lainnya.",
    "iui", 
    "IUI Mikro, Kecil, Menengah, dan Besar. Persyaratan umum: Fotokopi KTP, NPWP, akta pendirian usaha (untuk badan usaha), dokumen lingkungan (UKL-UPL atau AMDAL), dan dokumen pendukung lainnya.",
    "tdi", 
    "Untuk industri kecil dan menengah. Persyaratan: Fotokopi KTP, NPWP, dan dokumen pendukung lainnya.",
    "skdu", 
    "Persyaratan: Fotokopi KTP, NPWP, dan surat pernyataan domisili dari RT/RW setempat.",
    "proses_pengajuan_perizinan", 
    "Proses pengajuan perizinan meliputi pengumpulan berkas, pengajuan permohonan, verifikasi dan evaluasi, serta penerbitan izin jika dokumen dan persyaratan dinyatakan lengkap dan sesuai.",
    "website",
    "Website resmi dinas adalah [www.disperindagkotim.go.id](http://www.disperindagkotim.go.id)."
]

salam_questions = {
    "assalamualaikum": "Waalaikum sallam! Ada yang bisa saya bantu?",
    "halo": "Halo! Ada yang bisa saya bantu?",
    "hallo": "Hallo! Ada yang bisa saya bantu?",
    "hi": "Hallo! Ada yang bisa saya bantu?",
    "hai": "Hai! Ada yang bisa saya bantu?",
    "selamat pagi": "Selamat pagi! Ada yang bisa saya bantu?",
    "selamat siang": "Selamat siang! Ada yang bisa saya bantu?",
    "selamat sore": "Selamat sore! Ada yang bisa saya bantu?",
    "selamat malam": "Selamat malam! Ada yang bisa saya bantu?",
    "apa kabar": "Saya baik, terima kasih! Bagaimana dengan Anda?",
    "apa kabarmu": "Saya baik, terima kasih! Bagaimana dengan Anda?",
    "apa kabar kamu": "Saya baik, terima kasih! Bagaimana dengan Anda?",
    "terima kasih": "Sama-sama! Ada lagi yang bisa saya bantu?",
}

salam_angka = {
    "12 oke terima kasih" 
}
terima_kasih_responses = [
    "Sama-sama! Senang bisa membantu. Ada lagi yang bisa saya bantu?",
    "Terima kasih kembali! Jika ada pertanyaan lain, jangan ragu untuk menanyakan.",
    "Anda sangat sopan! Ada hal lain yang ingin Anda ketahui?",
    "Senang bisa membantu Anda! Apakah ada yang lain yang bisa saya bantu?",
    "Sama-sama! Apakah Anda memerlukan bantuan lain?",
    "Terima kasih kembali! Saya di sini jika Anda membutuhkan bantuan lebih lanjut.",
]

kepuasan_responses = [
    "Terima kasih atas feedback positifnya! Kami senang mendengar bahwa Anda puas dengan layanan kami.",
    "Senang mengetahui bahwa informasi kami bermanfaat untuk Anda!",
    "Kami menghargai feedback Anda! Terima kasih telah menggunakan layanan kami.",
    "Terima kasih atas pujiannya! Jangan ragu untuk menghubungi kami lagi jika Anda memerlukan bantuan lebih lanjut.",
    "Kami senang bisa membantu! Terima kasih atas feedbacknya.",
]

class DinasPerdagangan:
    def __init__(self):
        self.informasi = {
            # Masukkan informasi di sini seperti pada kode Anda
             "siup": "Untuk membuka usaha UMKM, Anda memerlukan SIUP Mikro atau Kecil. Persyaratan: Fotokopi KTP pemilik usaha, NPWP, akta pendirian usaha (untuk badan usaha), Surat Keterangan Domisili Usaha (SKDU) dari kelurahan/desa setempat, formulir permohonan SIUP yang sudah diisi.",
            "tdi": "Anda perlu mengurus TDI untuk UMKM. Persyaratan: Fotokopi KTP pemilik usaha, NPWP, formulir permohonan TDI yang sudah diisi, izin lingkungan (jika diperlukan, seperti UKL-UPL atau AMDAL).",
            "skdu": "Anda memerlukan SKDU untuk UMKM. Persyaratan: Fotokopi KTP pemilik usaha, NPWP, surat pernyataan domisili dari RT/RW setempat.",
            "nib": "Anda memerlukan NIB untuk UMKM sesuai dengan peraturan OSS (Online Single Submission). Persyaratan: Fotokopi KTP pemilik usaha, NPWP, akta pendirian usaha (untuk badan usaha), formulir permohonan NIB yang diisi melalui sistem OSS.",
            "proses_pengajuan_perizinan": "Proses pengajuan perizinan untuk UMKM meliputi: 1) Pengumpulan berkas: Kumpulkan semua dokumen persyaratan yang diperlukan sesuai jenis izin yang diajukan. 2) Pengajuan permohonan: Ajukan permohonan secara langsung ke kantor dinas atau melalui layanan online jika tersedia. 3) Verifikasi dan evaluasi: Petugas dinas akan melakukan verifikasi dan evaluasi terhadap dokumen yang diajukan. 4) Penerbitan izin: Jika dokumen dan persyaratan dinyatakan lengkap dan sesuai, dinas akan menerbitkan izin yang dimohonkan.",
            "kontak": "Telepon (0531) 21002, Email: disperindagkotim@example.com, Alamat: Jl. Jenderal Sudirman No. 45, Sampit, Kabupaten Kotawaringin Timur, Kalimantan Tengah.",
            "website": "Website resmi dinas adalah [www.disperindagkotim.go.id](http://www.disperindagkotim.go.id)."

        }

    def get_info(self, input_text):
        mapping = {
            # Masukkan mapping di sini seperti pada kode Anda
            "dinas: Bagaimana cara mengurus SIUP untuk UMKM?": "siup",
            "dinas: Bagaimana cara mengurus TDI untuk UMKM?": "tdi",
            "dinas: Apa saja persyaratan untuk mengurus SKDU untuk UMKM?": "skdu",
            "dinas: Bagaimana cara mengurus NIB untuk UMKM?": "nib",
            "dinas: Bagaimana proses pengajuan perizinan untuk UMKM?": "proses_pengajuan_perizinan",
            "dinas: Bagaimana cara menghubunginya?": "kontak",
            "dinas: Apa alamat website dinas?": "website"

        }
        return self.informasi.get(mapping.get(input_text.lower(), ""), "Maaf, saya tidak memiliki informasi yang Anda cari.")

@app.route('/')
def index():
    return render_template_string(open("templates/index.html").read())

@app.route('/ask', methods=['POST'])
def ask():
    inputan = request.json.get('question', '').strip().lower()
    
    app.logger.debug(f"Pertanyaan yang diterima: {inputan}")
    
    if inputan in ["keluar", "exit"]:
        return jsonify({"response": "Terima kasih! Sampai jumpa lagi."})
    
    if inputan in ["kapan", "bagaimana", "mengapa", "dimana", "siapa"]:
        return jsonify({"response": "pertanyaan anda tidak valid, tanyakan seputar Disperdagin."})
    
    if not inputan:
        return jsonify({"response": "Silakan masukkan pertanyaan yang valid."})

    if len(inputan) <= 1:
        return jsonify({"response": "Pertanyaan Anda terlalu pendek. Silakan masukkan pertanyaan yang lebih panjang."})

    # Cek salam
    if inputan in salam_questions:
        return jsonify({"response": salam_questions[inputan]})
    
    if inputan in salam_angka:
        return jsonify({"response": salam_angka[inputan]})
        
    # Menambahkan respon berdasarkan kata kunci
    if "terima kasih" in inputan:
        return jsonify({"response": random.choice(terima_kasih_responses)})
    
    if "terima kasih" in inputan:
        return jsonify({"response": random.choice(terima_kasih_responses)})
    
    if "informasi salah" in inputan or "informasi keliru" in inputan:
        return jsonify({"response": "Maaf atas ketidaknyamanannya. Silakan beri tahu informasi mana yang salah agar kami dapat memperbaikinya."})
    
    if "tidak puas" in inputan or "tidak membantu" in inputan:
        return jsonify({"response": "Kami mohon maaf jika jawaban kami tidak memuaskan. Silakan beri tahu pertanyaan lebih lanjut agar kami bisa membantu lebih baik."})

    # Menambahkan respon untuk ungkapan kepuasan
    if "puas" in inputan or "sangat membantu" in inputan:
        return jsonify({"response": random.choice(kepuasan_responses)})

    try:
        response = model.generate_content(prompt_parts + [inputan])
        return jsonify({"response": response.text})
    except Exception as e:
        app.logger.error(f"Terjadi kesalahan: {e}")
        return jsonify({"response": f"Maaf, terjadi kesalahan: {e}"})

if __name__ == '__main__':
    app.run(debug=True)
